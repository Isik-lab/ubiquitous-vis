"""
PC Interpretation: Predict movie features from PC time series.

This module implements cross-validated ridge regression to quantify how well
ROI PC spaces encode high-level movie features (social, valence, motion, etc.).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
import sys
import argparse
from datetime import datetime
from src import group_pca_analysis
from src import helpers


def create_chunk_labels(n_timepoints, chunklen=20):
    """
    Create chunk labels for GroupKFold CV.

    Args:
        n_timepoints: Total number of TRs (e.g., 1921)
        chunklen: Chunk length in TRs (default: 20)

    Returns:
        chunk_ids: (n_timepoints,) array of chunk IDs for GroupKFold
    """
    chunk_ids = np.arange(n_timepoints) // chunklen
    return chunk_ids


def block_permutation_null(pc_timeseries, label_values, speech_mask=None,
                           n_permutations=1000, block_size=20, random_state=42):
    """
    Circular-shift null for per-PC correlation r^2.

    Shifts the full-length label by a random offset in [block_size, T - block_size],
    then applies the speech mask (if any) to both PCs and shifted label, then
    computes Pearson r^2 per PC. Preserves within-series autocorrelation while
    destroying cross-series temporal alignment, so surviving r^2 is evidence
    against shared low-frequency stimulus structure alone.

    Args:
        pc_timeseries: (T, K) PC activations over time (pre-mask, full T)
        label_values: (T,) label values (pre-mask, full T)
        speech_mask: (T,) boolean mask applied after circular shift; None to skip
        n_permutations: number of circular-shift draws
        block_size: minimum absolute shift in TRs (excludes near-identity)
        random_state: seed for reproducibility

    Returns:
        null_r2: (n_permutations, K) null distribution of r^2 values
    """
    rng = np.random.default_rng(random_state)
    T, K = pc_timeseries.shape
    min_shift = int(block_size)
    max_shift = int(T - block_size)

    if speech_mask is not None:
        X_masked = pc_timeseries[speech_mask]
    else:
        X_masked = pc_timeseries

    # Pre-center and pre-normalize each PC column so null loop reduces to a dot product
    X_centered = X_masked - X_masked.mean(axis=0, keepdims=True)
    X_norm = np.linalg.norm(X_centered, axis=0)
    # Guard against zero-variance PCs (correlation undefined -> leave r = 0)
    X_norm_safe = np.where(X_norm > 0, X_norm, 1.0)

    null_r2 = np.zeros((n_permutations, K))

    for p in range(n_permutations):
        shift = int(rng.integers(min_shift, max_shift))
        rolled = np.roll(label_values, shift)
        if speech_mask is not None:
            y = rolled[speech_mask]
        else:
            y = rolled
        y_centered = y - y.mean()
        y_norm = np.linalg.norm(y_centered)
        if y_norm == 0:
            continue
        r = (X_centered.T @ y_centered) / (X_norm_safe * y_norm)
        r = np.where(X_norm > 0, r, 0.0)
        null_r2[p] = r ** 2

    return null_r2


def predict_label_per_pc(pc_timeseries, label_values, chunklen=20, alphas=None,
                        speech_mask=None, method='ridge',
                        n_permutations=0, block_size=20, perm_seed=42):
    """
    Fit separate models for each PC individually.

    Evaluates how well each PC independently relates to a label using either
    cross-validated ridge regression or simple correlation. This provides
    interpretable test-set performance for each orthogonal PC dimension.

    Args:
        pc_timeseries: (T, K) PC activations over time
        label_values: (T,) label values to predict
        chunklen: Chunk length for CV (default: 20 TRs)
        alphas: Ridge regularization grid (default: logspace(-2, 6, 25))
        speech_mask: (T,) boolean mask, if provided restricts analysis to speech periods
        method: 'ridge' for ridge regression or 'correlation' for Pearson correlation

    Returns:
        per_pc_results: List of K dicts, one per PC, each containing:
            - pc_index: PC number (0-indexed)
            - r2_mean: Mean cross-validated R² (ridge) or correlation (correlation)
            - r2_std: Standard deviation of R² or correlation
            - r2_folds: Per-fold R² scores or correlations
            - best_alpha: Selected regularization parameter (ridge only, None for correlation)
            - n_chunks: Number of chunks used
            - n_timepoints: Number of timepoints used
    """
    if alphas is None:
        alphas = np.logspace(-2, 6, 25)

    # Retain pre-mask copies for block-permutation null (which rolls on full T
    # then re-applies the speech mask, preserving natural autocorrelation).
    pc_full = pc_timeseries
    label_full = label_values

    # Apply speech mask if provided
    if speech_mask is not None:
        pc_timeseries = pc_timeseries[speech_mask]
        label_values = label_values[speech_mask]

    T, K = pc_timeseries.shape
    per_pc_results = []

    if method == 'correlation':
        # Optional block-permutation null (shared across PCs within this label)
        null_r2 = None
        if n_permutations and n_permutations > 0:
            null_r2 = block_permutation_null(
                pc_full, label_full, speech_mask=speech_mask,
                n_permutations=n_permutations, block_size=block_size,
                random_state=perm_seed
            )

        # Simple correlation for each PC
        for k in range(K):
            pc_k = pc_timeseries[:, k]

            # Compute correlation
            r, p = pearsonr(pc_k, label_values)
            r2 = r ** 2

            pc_entry = {
                'pc_index': k,
                'r2_mean': r2,
                'r2_std': 0.0,  # No CV for simple correlation
                'r2_folds': [r2],
                'correlation': r,
                'p_value': p,
                'best_alpha': None,
                'n_chunks': 0,
                'n_timepoints': T
            }

            if null_r2 is not None:
                null_k = null_r2[:, k]
                p_perm = (1 + np.sum(null_k >= r2)) / (n_permutations + 1)
                pc_entry['p_value_perm'] = p_perm
                pc_entry['null_r2_mean'] = float(null_k.mean())
                pc_entry['null_r2_std'] = float(null_k.std())
                pc_entry['null_r2_95'] = float(np.percentile(null_k, 95))
                pc_entry['n_permutations'] = n_permutations

            per_pc_results.append(pc_entry)

    elif method == 'ridge':
        # Fit separate ridge model for each PC
        for k in range(K):
            # Extract single PC, keep 2D shape (T, 1)
            pc_k = pc_timeseries[:, k:k+1]

            # Standardize
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X = scaler_X.fit_transform(pc_k)
            y = scaler_y.fit_transform(label_values.reshape(-1, 1)).squeeze()

            # Create chunk labels
            chunk_ids = create_chunk_labels(T, chunklen)
            n_chunks = len(np.unique(chunk_ids))

            # 5-fold grouped CV
            gkf = GroupKFold(n_splits=5)

            # Alpha selection
            ridge = RidgeCV(
                alphas=alphas,
                cv=gkf.split(X, y, groups=chunk_ids),
                scoring='r2'
            )
            ridge.fit(X, y)

            # Compute per-fold R² scores
            r2_scores = []
            for train_idx, test_idx in gkf.split(X, y, groups=chunk_ids):
                ridge_fold = Ridge(alpha=ridge.alpha_)
                ridge_fold.fit(X[train_idx], y[train_idx])
                r2 = ridge_fold.score(X[test_idx], y[test_idx])
                r2_scores.append(r2)

            per_pc_results.append({
                'pc_index': k,
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
                'r2_folds': r2_scores,
                'best_alpha': ridge.alpha_,
                'n_chunks': n_chunks,
                'n_timepoints': T
            })

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ridge' or 'correlation'")

    return per_pc_results


def predict_label_from_pcs(pc_timeseries, label_values, chunklen=20, alphas=None,
                          speech_mask=None, fit_per_pc=False, method='ridge',
                          n_permutations=0, block_size=20, perm_seed=42):
    """
    Relate PC time series to a label using ridge regression or correlation.

    Predicts a single label from PC time series using either cross-validated
    ridge regression (default) or simple Pearson correlation.

    Args:
        pc_timeseries: (T, K) PC activations over time
        label_values: (T,) label values to predict
        chunklen: Chunk length for CV (default: 20 TRs, matches encoding model)
        alphas: Ridge regularization grid (default: logspace(-2, 6, 25))
        speech_mask: (T,) boolean mask, if provided restricts analysis to speech periods
        fit_per_pc: If True, also fit separate models for each PC (default: False)
        method: 'ridge' for ridge regression or 'correlation' for Pearson correlation

    Returns:
        results: Dict with:
            - r2_mean: Mean cross-validated R² (ridge) or mean R² from correlations (correlation)
            - r2_std: Standard deviation of R² (ridge) or std of R² from correlations (correlation)
            - r2_folds: Per-fold R² scores (ridge) or per-PC R² values (correlation)
            - best_alpha: Selected regularization parameter (ridge only, None for correlation)
            - beta_weights: (K,) regression coefficients (ridge) or correlations (correlation)
            - n_chunks: Number of chunks used in CV (ridge only, 0 for correlation)
            - n_timepoints: Number of timepoints used (after masking)
            - per_pc_results: (optional) List of dicts with per-PC results if fit_per_pc=True
    """
    if alphas is None:
        alphas = np.logspace(-2, 6, 25)

    # Retain pre-mask copies so per-PC null (if requested) can roll on full T
    # before masking (preserves natural label autocorrelation).
    pc_full = pc_timeseries
    label_full = label_values

    # Apply speech mask if provided
    if speech_mask is not None:
        pc_timeseries = pc_timeseries[speech_mask]
        label_values = label_values[speech_mask]
        print(f"  Applied speech mask: {len(pc_timeseries)} / {len(speech_mask)} TRs retained")

    T, K = pc_timeseries.shape

    if method == 'correlation':
        # Simple correlation for each PC
        correlations = []
        r2_values = []
        p_values = []

        for k in range(K):
            r, p = pearsonr(pc_timeseries[:, k], label_values)
            correlations.append(r)
            r2_values.append(r ** 2)
            p_values.append(p)

        results = {
            'r2_mean': np.mean(r2_values),
            'r2_std': np.std(r2_values),
            'r2_folds': r2_values,  # Per-PC R² values
            'best_alpha': None,
            'beta_weights': np.array(correlations),  # Store correlations
            'correlations': np.array(correlations),
            'p_values': np.array(p_values),
            'n_chunks': 0,
            'n_timepoints': T
        }

    elif method == 'ridge':
        # Standardize (z-score)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X = scaler_X.fit_transform(pc_timeseries)
        y = scaler_y.fit_transform(label_values.reshape(-1, 1)).squeeze()

        # Create chunk labels for GroupKFold
        chunk_ids = create_chunk_labels(T, chunklen)
        n_chunks = len(np.unique(chunk_ids))

        # 5-fold grouped CV (chunks never split across train/test)
        gkf = GroupKFold(n_splits=5)

        # Alpha selection with grouped CV
        ridge = RidgeCV(
            alphas=alphas,
            cv=gkf.split(X, y, groups=chunk_ids),
            scoring='r2'
        )
        ridge.fit(X, y)

        # Compute per-fold R² scores with best alpha
        r2_scores = []

        for train_idx, test_idx in gkf.split(X, y, groups=chunk_ids):
            ridge_fold = Ridge(alpha=ridge.alpha_)
            ridge_fold.fit(X[train_idx], y[train_idx])

            # Evaluate on test fold
            r2 = ridge_fold.score(X[test_idx], y[test_idx])
            r2_scores.append(r2)

        results = {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'r2_folds': r2_scores,
            'best_alpha': ridge.alpha_,
            'beta_weights': ridge.coef_,  # (K,) for interpretation
            'n_chunks': n_chunks,
            'n_timepoints': T
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ridge' or 'correlation'")

    # Optionally fit separate models for each PC. Pass pre-mask data + mask so
    # the block-permutation null rolls on the full T before masking.
    if fit_per_pc:
        print("  Fitting individual PC models...")
        per_pc_results = predict_label_per_pc(
            pc_full, label_full, chunklen, alphas, speech_mask, method,
            n_permutations=n_permutations, block_size=block_size,
            perm_seed=perm_seed
        )
        results['per_pc_results'] = per_pc_results

    return results


def load_speech_mask(data_dir):
    """
    Load speech mask from speaking.csv.

    Args:
        data_dir: Project root directory containing features/

    Returns:
        speech_mask: (1921,) boolean array, True during speech periods
    """
    speaking_path = os.path.join(data_dir, 'features', 'speaking.csv')

    if not os.path.exists(speaking_path):
        raise FileNotFoundError(f"Speech mask not found: {speaking_path}")

    speaking_vec = pd.read_csv(speaking_path, header=None).values.squeeze()
    speech_mask = speaking_vec > 0

    print(f"Loaded speech mask: {speech_mask.sum()} / {len(speech_mask)} TRs ({100 * speech_mask.mean():.1f}%)")

    return speech_mask


def load_movie_labels(data_dir, label_names=None):
    """
    Load labeled movie features aligned to encoding model timing.

    Args:
        data_dir: Project root directory containing features/ subdirectory
        label_names: List of feature names to load. If None, load all available.

    Returns:
        labels_df: DataFrame with columns = label_names, rows = 1921 TRs
    """
    if label_names is None:
        # Default: all available labels
        label_names = [
            'social', 'num_agents', 'turn_taking', 'mentalization',
            'valence', 'arousal',
            'motion', 'face', 'hue', 'pixel',
            'speaking', 'music', 'amplitude', 'pitch',
            'written_text'
        ]

    labels_df = pd.DataFrame()

    for label in label_names:
        filepath = os.path.join(data_dir, 'features', f'{label}.csv')

        if os.path.exists(filepath):
            label_data = pd.read_csv(filepath, header=None).values.squeeze()
            labels_df[label] = label_data
            print(f"Loaded {label}: shape {label_data.shape}")
        else:
            print(f"Warning: {label}.csv not found at {filepath}")

    print(f"\nLoaded {len(labels_df.columns)} labels with {len(labels_df)} timepoints")

    return labels_df


def predict_all_labels_from_pcs(pc_timeseries, labels_df, chunklen=20, alphas=None,
                               speech_mask=None, fit_per_pc=False, method='ridge',
                               n_permutations=0, block_size=20, perm_seed=42):
    """
    Predict all labels from PC time series.

    Args:
        pc_timeseries: (T, K) PC activations
        labels_df: DataFrame with label columns
        chunklen: Chunk length for CV
        alphas: Ridge regularization grid
        speech_mask: (T,) boolean mask for speech periods
        fit_per_pc: If True, also fit separate models for each PC (default: False)
        method: 'ridge' for ridge regression or 'correlation' for Pearson correlation

    Returns:
        results_dict: {label_name: prediction_results}
    """
    results_dict = {}

    for label in labels_df.columns:
        print(f"  Predicting {label}...")
        results = predict_label_from_pcs(
            pc_timeseries,
            labels_df[label].values,
            chunklen=chunklen,
            alphas=alphas,
            speech_mask=speech_mask,
            fit_per_pc=fit_per_pc,
            method=method,
            n_permutations=n_permutations,
            block_size=block_size,
            perm_seed=perm_seed
        )
        results_dict[label] = results

        if method == 'correlation':
            print(f"    Mean R² = {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        else:
            print(f"    Combined model R² = {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")

        # Print per-PC results if available
        if fit_per_pc and 'per_pc_results' in results:
            for pc_result in results['per_pc_results']:
                pc_idx = pc_result['pc_index']
                if method == 'correlation':
                    corr = pc_result.get('correlation', np.sqrt(pc_result['r2_mean']))
                    print(f"      PC{pc_idx+1} r = {corr:.4f}, R² = {pc_result['r2_mean']:.4f}")
                else:
                    print(f"      PC{pc_idx+1} alone R² = {pc_result['r2_mean']:.4f} ± {pc_result['r2_std']:.4f}")

    return results_dict


def compile_pooled_results(results_dir, output_file=None):
    """
    Combine the per-combination pooled PC interpretation CSVs into one table.

    Drops speaking and adds ``q_value_fdr`` (see add_fdr): Benjamini-Hochberg FDR on
    ``p_value_perm`` within each (feature_space, roi, roi_type) panel (10 PCs x 14 labels),
    the family of comparisons shown in each panel of the pooled PCA summary figure.

    Args:
        results_dir: Directory of ``pooled_*.csv`` files written by
            python -m src.pc_interpretation run --use_pooled_pca --fit_per_pc
        output_file: Path to save the compiled CSV (optional)

    Returns:
        compiled_df: DataFrame with one row per (feature_space, roi, PC, label)
    """
    csv_files = sorted(Path(results_dir).glob('pooled_*.csv'))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No pooled_*.csv result files found in {results_dir}")
    print(f"Compiling {len(csv_files)} pooled PC interpretation files from {results_dir}")

    compiled_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    compiled_df = add_fdr(compiled_df)

    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        compiled_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

    return compiled_df


def add_fdr(compiled_df, exclude_labels=('speaking',)):
    """
    Drop ``exclude_labels`` and add ``q_value_fdr`` (BH-FDR on ``p_value_perm``
    within each (feature_space, roi, roi_type) panel) to compiled pooled results.

    speaking is excluded because the analysis is restricted to speech, where it is
    constant: its correlation is undefined, but its permutation p-value is the
    minimum, which would otherwise enter (and loosen) the FDR correction.
    """
    from statsmodels.stats.multitest import multipletests

    compiled_df = compiled_df[~compiled_df['label'].isin(exclude_labels)].copy()

    compiled_df['q_value_fdr'] = np.nan
    for _, group_df in compiled_df.groupby(['feature_space', 'roi', 'roi_type']):
        pvals = group_df['p_value_perm'].values
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            continue
        q = np.full(len(pvals), np.nan)
        q[valid] = multipletests(pvals[valid], method='fdr_bh')[1]
        compiled_df.loc[group_df.index, 'q_value_fdr'] = q
    print(f"  {(compiled_df['q_value_fdr'] < 0.05).sum():,} / {len(compiled_df):,} entries q<0.05")

    col_order = ['subject', 'feature_space', 'roi_type', 'roi', 'pc_index', 'label']
    compiled_df = compiled_df[col_order + [c for c in compiled_df.columns if c not in col_order]]
    compiled_df = compiled_df.sort_values(
        ['feature_space', 'roi_type', 'roi', 'pc_index', 'label']).reset_index(drop=True)

    return compiled_df


# ============================================================================
# Command line: run one (pooled) PC interpretation, or compile all results
#   python -m src.pc_interpretation run --subject pooled --feature_space motion --roi MT ...
#   python -m src.pc_interpretation compile
# ============================================================================

def analyze_subject_roi(subject, feature_space, roi, roi_type, args):
    """
    Run PC interpretation for one subject/ROI/feature combination.

    Args:
        subject: e.g., 'sub-05'
        feature_space: e.g., 'alexnet_layer6'
        roi: e.g., 'MT'
        roi_type: 'func' or 'anat'
        args: Namespace with configuration

    Returns:
        dict with status and results
    """
    print(f"\n{'='*60}")
    print(f"PC Interpretation: {subject} / {feature_space} / {roi_type} / {roi}")
    print(f"{'='*60}")

    try:
        # 1. Load PC loadings (pooled or per-subject)
        print("\n1. Loading PC loadings...")
        if args.use_pooled_pca:
            # Load shared PC loadings from pooled PCA
            print("   Using POOLED PCA (shared dimensions across all subjects)")
            pcs = group_pca_analysis.load_pooled_pc_loadings(
                feature_space, roi, args.data_dir,
                n_components=args.n_components,
                roi_type=roi_type,
                model=args.model,
                population=args.population
            )
            print(f"   Loaded {pcs['n_components']} pooled PCs, {pcs['n_features']} features")
            print(f"   Based on {pcs['metadata']['n_subjects']} subjects, "
                  f"{pcs['metadata']['total_voxels']} total voxels")
        else:
            # Load per-subject PC loadings (original approach)
            print("   Using PER-SUBJECT PCA (individual dimensions)")
            pcs = group_pca_analysis.load_subject_pc_loadings(
                subject, feature_space, roi, args.data_dir,
                n_components=args.n_components,
                roi_type=roi_type
            )
            print(f"   Loaded {pcs['n_components']} PCs, {pcs['n_features']} features")

        # 2. Load movie embeddings (concatenate across feature spaces if pooled PCA used multiple)
        print("\n2. Loading movie embeddings...")
        if args.use_pooled_pca:
            feature_spaces_list = pcs['feature_spaces']
            print(f"   Pooled PCA feature spaces: {feature_spaces_list}")
            embedding_blocks = [
                group_pca_analysis.load_movie_embeddings(fs, args.data_dir)
                for fs in feature_spaces_list
            ]
            movie_embeddings = np.concatenate(embedding_blocks, axis=1)
            if movie_embeddings.shape[1] != pcs['n_features']:
                raise ValueError(
                    f"Concatenated embedding dim {movie_embeddings.shape[1]} does not match "
                    f"pooled PC feature dim {pcs['n_features']}. "
                    f"Feature spaces (in order): {feature_spaces_list}"
                )
            print(f"   Concatenated movie embeddings: shape {movie_embeddings.shape}")
        else:
            movie_embeddings = group_pca_analysis.load_movie_embeddings(
                feature_space, args.data_dir
            )

        # 3. Project movie to PC space
        print("\n3. Projecting movie onto PC space...")
        pc_timeseries = group_pca_analysis.project_movie_to_pcs(
            movie_embeddings, pcs['loadings']
        )
        print(f"   PC timeseries: {pc_timeseries.shape}")

        # 4. Load movie labels
        print("\n4. Loading movie labels...")
        labels_df = load_movie_labels(
            args.data_dir, label_names=args.labels
        )
        if args.use_speech_mask:
            # speaking is constant during speech, so it cannot be correlated with the PCs
            labels_df = labels_df.drop(columns='speaking', errors='ignore')

        # 5. Load speech mask if requested
        speech_mask = None
        if args.use_speech_mask:
            print("\n5. Loading speech mask...")
            speech_mask = load_speech_mask(args.data_dir)

        # 6. Predict all labels
        print(f"\n{'6' if not args.use_speech_mask else '6'}. Predicting labels from PCs...")
        results_dict = predict_all_labels_from_pcs(
            pc_timeseries,
            labels_df,
            chunklen=args.chunklen,
            speech_mask=speech_mask,
            fit_per_pc=args.fit_per_pc,
            method=args.method,
            n_permutations=args.n_permutations,
            block_size=args.block_size,
            perm_seed=args.perm_seed
        )

        # 7. Save results
        print("\n7. Saving results...")
        save_interpretation_results(
            subject, feature_space, roi, roi_type,
            results_dict, labels_df.columns, args
        )

        print(f"\n{'='*60}")
        print("SUCCESS")
        print(f"{'='*60}\n")

        return {
            'status': 'success',
            'n_components': pcs['n_components'],
            'n_features': pcs['n_features'],
            'n_labels': len(results_dict),
            'results': results_dict
        }

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'status': 'failed',
            'error': str(e)
        }


def save_interpretation_results(subject, feature_space, roi, roi_type,
                               results_dict, label_names, args):
    """Save PC interpretation results to CSV files."""

    # Create output directories
    out_base = os.path.join(args.out_dir, 'PCInterpretation')

    # Use different directories for pooled vs per-subject results
    if args.use_pooled_pca:
        results_dir = os.path.join(out_base, 'pooled_results')
        beta_dir = os.path.join(out_base, 'pooled_beta_weights')
        summary_dir = os.path.join(out_base, 'pooled_summary')
        per_pc_dir = os.path.join(out_base, 'pooled_per_pc_results')
    else:
        results_dir = os.path.join(out_base, 'results')
        beta_dir = os.path.join(out_base, 'beta_weights')
        summary_dir = os.path.join(out_base, 'summary')
        per_pc_dir = os.path.join(out_base, 'per_pc_results')

    for d in [results_dir, beta_dir, summary_dir, per_pc_dir]:
        os.makedirs(d, exist_ok=True)

    # File label
    if args.use_pooled_pca:
        label = f"{subject}_{group_pca_analysis.pooled_label(feature_space, roi_type, roi, args.model, args.population)}"
    else:
        label = f"{subject}_{feature_space}_{roi_type}_{roi}"

    # 1. Save summary results (R² scores for all labels)
    results_rows = []
    for label_name in label_names:
        if label_name in results_dict:
            res = results_dict[label_name]
            results_rows.append({
                'subject': subject,
                'feature_space': feature_space,
                'roi': roi,
                'roi_type': roi_type,
                'label': label_name,
                'r2_mean': res['r2_mean'],
                'r2_std': res['r2_std'],
                'best_alpha': res['best_alpha'],
                'n_components': res['beta_weights'].shape[0],
                'n_chunks': res['n_chunks'],
                'n_timepoints': res['n_timepoints']
            })

    results_df = pd.DataFrame(results_rows)
    results_file = os.path.join(results_dir, f"{label}_interpretation_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"   Saved results: {results_file}")

    # 2. Save beta weights for each label
    for label_name in label_names:
        if label_name in results_dict:
            res = results_dict[label_name]
            beta_weights = res['beta_weights']

            # Create DataFrame: rows = PCs, columns = ['PC', 'beta_weight']
            beta_dict = {
                'PC': [f'PC{i+1}' for i in range(len(beta_weights))],
                'beta_weight': beta_weights
            }

            # Add correlation and p-values if available
            if 'correlations' in res:
                beta_dict['correlation'] = res['correlations']
            if 'p_values' in res:
                beta_dict['p_value'] = res['p_values']

            beta_df = pd.DataFrame(beta_dict)

            beta_file = os.path.join(beta_dir, f"{label}_{label_name}_beta_weights.csv")
            beta_df.to_csv(beta_file, index=False)

    print(f"   Saved beta weights: {len(label_names)} files")

    # 3. Save per-PC results if available
    per_pc_rows = []
    for label_name in label_names:
        if label_name in results_dict and 'per_pc_results' in results_dict[label_name]:
            for pc_result in results_dict[label_name]['per_pc_results']:
                row = {
                    'subject': subject,
                    'feature_space': feature_space,
                    'roi': roi,
                    'roi_type': roi_type,
                    'label': label_name,
                    'pc_index': pc_result['pc_index'],
                    'r2_mean': pc_result['r2_mean'],
                    'r2_std': pc_result['r2_std'],
                    'best_alpha': pc_result['best_alpha'],
                    'n_chunks': pc_result['n_chunks'],
                    'n_timepoints': pc_result['n_timepoints']
                }
                # Add correlation and p-value if available
                if 'correlation' in pc_result:
                    row['correlation'] = pc_result['correlation']
                if 'p_value' in pc_result:
                    row['p_value'] = pc_result['p_value']
                for key in ('p_value_perm', 'null_r2_mean', 'null_r2_std',
                            'null_r2_95', 'n_permutations'):
                    if key in pc_result:
                        row[key] = pc_result[key]
                per_pc_rows.append(row)

    if per_pc_rows:
        per_pc_df = pd.DataFrame(per_pc_rows)
        per_pc_file = os.path.join(per_pc_dir, f"{label}_per_pc_results.csv")
        per_pc_df.to_csv(per_pc_file, index=False)
        print(f"   Saved per-PC results: {per_pc_file}")

    # 4. Append to summary log
    log_file = os.path.join(summary_dir, 'interpretation_log.csv')
    log_entry = {
        'subject': subject,
        'feature_space': feature_space,
        'roi': roi,
        'roi_type': roi_type,
        'n_labels': len(results_dict),
        'mean_r2': np.mean([res['r2_mean'] for res in results_dict.values()]),
        'max_r2': np.max([res['r2_mean'] for res in results_dict.values()]),
        'status': 'success',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    log_df = pd.DataFrame([log_entry])

    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    print(f"   Updated log: {log_file}")


def run_main():
    parser = argparse.ArgumentParser(
        description='Run PC interpretation analysis'
    )

    # Paths
    parser.add_argument('--dir', default=helpers.PROJECT_DIR,
                       help='Project root directory')
    parser.add_argument('--data_dir', default=None,
                       help='Data directory (defaults to --dir)')
    parser.add_argument('--out_dir', default='analysis',
                       help='Output directory (relative to --dir)')

    # Analysis specification
    parser.add_argument('--subject', type=str,
                       help='Subject ID (e.g., sub-05)')
    parser.add_argument('--feature_space', type=str,
                       help='Feature space (e.g., alexnet_layer6)')
    parser.add_argument('--roi', type=str,
                       help='ROI name (e.g., MT)')
    parser.add_argument('--roi_type', default='func', choices=['func', 'anat'],
                       help='ROI type: func or anat')

    # Batch mode
    parser.add_argument('--batch', action='store_true',
                       help='Run all successful combinations from WeightPCA')
    parser.add_argument('--combination_index', type=int,
                       help='Process specific row from combinations.tsv (1-indexed)')

    # Parameters
    parser.add_argument('--n_components', type=int, default=10,
                       help='Number of PCs to use')
    parser.add_argument('--chunklen', type=int, default=20,
                       help='Chunk length for CV')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='Specific labels to predict (default: all)')
    parser.add_argument('--use_speech_mask', action='store_true',
                       help='Restrict analysis to speech periods')
    parser.add_argument('--fit_per_pc', action='store_true',
                       help='Fit separate models for each PC individually')
    parser.add_argument('--method', type=str, default='correlation', choices=['ridge', 'correlation'],
                       help='Method for relating PCs to features: ridge or correlation')
    parser.add_argument('--n_permutations', type=int, default=10000,
                       help='Block-permutation null draws for per-PC r^2 significance (0 disables)')
    parser.add_argument('--block_size', type=int, default=120,
                       help='Minimum absolute circular-shift in TRs for block permutation '
                            '(should match encoding CV chunklen to respect temporal-independence assumption)')
    parser.add_argument('--perm_seed', type=int, default=42,
                       help='Random seed for block-permutation null')
    parser.add_argument('--use_pooled_pca', action='store_true',
                       help='Use pooled PCA (shared dimensions across subjects) instead of per-subject PCA')
    parser.add_argument('--model', default='vislang_gpt2_chunk60',
                       help='Encoding model name (used in pooled PCA filenames)')
    parser.add_argument('--population', default='NT',
                       help='Population name (NT)')

    args = parser.parse_args()

    # Resolve paths
    if args.data_dir is None:
        args.data_dir = args.dir
    args.out_dir = os.path.join(args.dir, args.out_dir)

    # Handle combination_index (for array jobs)
    if args.combination_index is not None:
        combinations_file = os.path.join(args.out_dir, 'PCInterpretation', 'combinations.tsv')

        if not os.path.exists(combinations_file):
            print(f"ERROR: Combinations file not found: {combinations_file}")
            print("Run generate_pc_interpretation_combinations.py first.")
            sys.exit(1)

        combinations = pd.read_csv(combinations_file, sep='\t')

        if args.combination_index < 1 or args.combination_index > len(combinations):
            print(f"ERROR: Invalid combination_index {args.combination_index}")
            print(f"Valid range: 1 to {len(combinations)}")
            sys.exit(1)

        # Get combination (1-indexed)
        combo = combinations.iloc[args.combination_index - 1]
        args.subject = combo['subject']
        args.feature_space = combo['feature_space']
        args.roi = combo['roi']
        args.roi_type = combo['roi_type']

        print(f"Processing combination {args.combination_index}/{len(combinations)}")

    # Check required arguments
    if not args.batch and not args.combination_index:
        if not all([args.subject, args.feature_space, args.roi]):
            parser.error("Either --batch, --combination_index, or all of (--subject, --feature_space, --roi) required")

    # Run analysis
    if args.batch:
        print("ERROR: Batch mode not yet implemented. Use --combination_index with array jobs.")
        sys.exit(1)
    else:
        result = analyze_subject_roi(
            args.subject, args.feature_space, args.roi, args.roi_type, args
        )

        if result['status'] != 'success':
            sys.exit(1)


def compile_main():
    parser = argparse.ArgumentParser(description='Compile pooled PC interpretation results into one CSV')
    parser.add_argument('--dir', default=helpers.PROJECT_DIR,
                        help='Project root directory')
    args = parser.parse_args()

    out_base = os.path.join(args.dir, 'analysis', 'PCInterpretation')
    compile_pooled_results(
        os.path.join(out_base, 'pooled_per_pc_results'),
        output_file=os.path.join(out_base, 'compiled_pooled_results.csv'),
    )


if __name__ == '__main__':
    commands = {'run': run_main, 'compile': compile_main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit('usage: python -m ' + __spec__.name + ' {run|compile} [options]')
    commands[sys.argv.pop(1)]()
