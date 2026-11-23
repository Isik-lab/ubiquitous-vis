"""
Group PCA Analysis: Align PC spaces across subjects using Procrustes.

This module handles loading pre-computed PCs from WeightPCA analysis,
aligning them across subjects for group visualization, and projecting
movie embeddings onto PC spaces.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from scipy.spatial import procrustes

# Linux caps a single filename component at 255 bytes. When many feature
# spaces are concatenated (e.g., SimCLR attention+embedding top-3), the raw
# feature_space string alone can blow that budget once the suffix and
# extension are appended. `pooled_label` produces a deterministic, filesystem-
# safe label that both the writer and readers can reconstruct from the same
# inputs — the full feature_space is still preserved in the HDF5 attrs.
_MAX_FEATURE_LABEL_LEN = 100


def _shorten_feature_space(feature_space):
    if len(feature_space) <= _MAX_FEATURE_LABEL_LEN:
        return feature_space
    # Keep the first component intact so the file is still recognizable at a
    # glance; append a stable short hash of the full string for uniqueness.
    first = feature_space.split('+', 1)[0][:60]
    digest = hashlib.sha1(feature_space.encode()).hexdigest()[:10]
    return f"{first}__h{digest}"


def pooled_label(feature_space, roi_type, roi, model, population):
    """Return the on-disk label used for pooled-PCA outputs."""
    return f"{_shorten_feature_space(feature_space)}_{roi_type}_{roi}_{model}_{population}_pooled"


def load_subject_pc_loadings(subject, feature_space, roi, data_dir, n_components=10, roi_type='func'):
    """
    Load pre-computed PC loadings from WeightPCA analysis.

    Args:
        subject: e.g., 'sub-05'
        feature_space: 'alexnet_layer6' or 'sbert_layer10'
        roi: 'MT', 'STS', 'temporal_language', 'frontal'
        data_dir: Project root directory
        n_components: Number of PCs to load (default: 10)
        roi_type: 'func' or 'anat' (default: 'func')

    Returns:
        dict with:
            - loadings: (K, D) PC loadings in feature space
            - explained_var: (K,) variance explained ratios
            - n_components: K (actual number loaded)
    """
    # New format with roi_type
    loadings_file = os.path.join(
        data_dir, 'analysis', 'WeightPCA', 'loadings',
        f'{subject}_{feature_space}_{roi_type}_{roi}_pc_loadings.csv'
    )

    # Backward compatibility: try old format if new format doesn't exist
    if not os.path.exists(loadings_file) and roi_type == 'func':
        loadings_file = os.path.join(
            data_dir, 'analysis', 'WeightPCA', 'loadings',
            f'{subject}_{feature_space}_{roi}_pc_loadings.csv'
        )

    variance_file = os.path.join(
        data_dir, 'analysis', 'WeightPCA', 'variance',
        f'{subject}_{feature_space}_{roi_type}_{roi}_variance.csv'
    )

    # Backward compatibility for variance file
    if not os.path.exists(variance_file) and roi_type == 'func':
        variance_file = os.path.join(
            data_dir, 'analysis', 'WeightPCA', 'variance',
            f'{subject}_{feature_space}_{roi}_variance.csv'
        )

    if not os.path.exists(loadings_file):
        raise FileNotFoundError(f"PC loadings not found: {loadings_file}")

    # Load PC loadings (rows = feature dims, columns = PCs)
    loadings_df = pd.read_csv(loadings_file)

    # First column is 'feature_dim', rest are PC1, PC2, etc.
    pc_columns = [col for col in loadings_df.columns if col.startswith('PC')]
    K = min(n_components, len(pc_columns))

    # Extract first K PCs: (n_features, K) then transpose to (K, n_features)
    loadings = loadings_df[pc_columns[:K]].values.T  # (K, D)

    # Load variance explained
    if os.path.exists(variance_file):
        variance_df = pd.read_csv(variance_file)
        variance_explained = variance_df['variance_explained'].values[:K]
    else:
        variance_explained = None
        print(f"Warning: Variance file not found: {variance_file}")

    return {
        'loadings': loadings,
        'explained_var': variance_explained,
        'n_components': K,
        'n_features': loadings.shape[1]
    }


def load_pooled_pc_loadings(feature_space, roi, data_dir, n_components=10, roi_type='func',
                            model='vislang_gpt2_chunk60', population='NT'):
    """
    Load pre-computed pooled (shared) PC loadings from PooledPCA analysis.

    These are PCs computed on pooled voxels from all subjects, representing
    shared semantic dimensions across the population.

    Args:
        feature_space: 'alexnet_layer6' or 'sbert_layer10'
        roi: 'MT', 'STS', 'temporal_language', 'frontal'
        data_dir: Project root directory
        n_components: Number of PCs to load (default: 10)
        roi_type: 'func' or 'anat' (default: 'func')
        model: Encoding model name (default: 'vislang_gpt2_chunk60')
        population: Population name (default: 'NT')

    Returns:
        dict with:
            - loadings: (K, D) shared PC loadings in feature space
            - explained_var: (K,) variance explained ratios
            - n_components: K (actual number loaded)
            - n_features: D (feature dimensionality)
            - metadata: dict with additional info (subjects, total_voxels, etc.)
    """
    import h5py
    import json

    label = pooled_label(feature_space, roi_type, roi, model, population)

    # Load PC loadings from HDF5
    loadings_file = os.path.join(
        data_dir, 'analysis', 'PooledPCA', 'loadings',
        f'{label}_pc_loadings.h5'
    )

    if not os.path.exists(loadings_file):
        raise FileNotFoundError(
            f"Pooled PC loadings not found: {loadings_file}\n"
            f"Run pooled PCA analysis first using python -m src.pooled_pca run"
        )

    with h5py.File(loadings_file, 'r') as f:
        all_loadings = f['loadings'][()]  # (K_total, D)
        all_explained_var = f['explained_var'][()]
        all_cumulative_var = f['cumulative_variance'][()]

        # Truncate to requested n_components
        K_total = all_loadings.shape[0]
        K = min(n_components, K_total)

        loadings = all_loadings[:K, :]  # (K, D)
        explained_var = all_explained_var[:K]
        cumulative_var = all_cumulative_var[:K]

        # Load metadata from attributes
        feature_spaces_attr = f.attrs.get('feature_spaces', None)
        if feature_spaces_attr is not None:
            feature_spaces_list = [s for s in str(feature_spaces_attr).split(',') if s]
        else:
            feature_spaces_list = [str(f.attrs['feature_space'])]

        metadata = {
            'n_components_total': int(f.attrs['n_components']),
            'n_features': int(f.attrs['n_features']),
            'total_voxels': int(f.attrs['total_voxels']),
            'n_subjects': int(f.attrs['n_subjects']),
            'feature_space': str(f.attrs['feature_space']),
            'feature_spaces': feature_spaces_list,
            'roi': str(f.attrs['roi']),
            'roi_type': str(f.attrs['roi_type']),
            'date': str(f.attrs['date'])
        }

    # Load summary file for subject list and voxel counts (optional)
    summary_file = os.path.join(
        data_dir, 'analysis', 'PooledPCA', 'summary',
        f'{label}_summary.json'
    )

    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                metadata['subjects'] = summary.get('subjects', [])
                metadata['n_voxels_per_subject'] = summary.get('n_voxels_per_subject', {})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Could not load summary file (continuing without it): {e}")
            metadata['subjects'] = []
            metadata['n_voxels_per_subject'] = {}

    print(f"Loaded pooled PCs: {feature_space} | {roi_type} {roi}")
    print(f"  Components: {K} / {K_total} (requested / available)")
    print(f"  Features: {loadings.shape[1]}")
    print(f"  Subjects: {metadata['n_subjects']}")
    print(f"  Total voxels: {metadata['total_voxels']}")
    print(f"  Cumulative variance: {cumulative_var[-1]:.4f}")

    return {
        'loadings': loadings,
        'explained_var': explained_var,
        'cumulative_variance': cumulative_var,
        'n_components': K,
        'n_features': loadings.shape[1],
        'feature_spaces': metadata['feature_spaces'],
        'metadata': metadata
    }


def procrustes_align_pcs(subject_loadings_dict, reference_subject=None):
    """
    Align all subjects' PC loadings to reference using Procrustes rotation.

    Handles cases where subjects have different numbers of PCs by truncating
    all subjects to the minimum number of PCs available across subjects.

    Args:
        subject_loadings_dict: {subject: (K, D) loadings array}
        reference_subject: Subject ID to use as reference. If None, use first subject.

    Returns:
        dict with:
            - aligned_loadings: {subject: (K, D) aligned loadings}
            - group_loadings: (K, D) mean of aligned loadings
            - disparities: {subject: disparity value}
            - reference_subject: Which subject was used as reference
            - n_components_used: K (number of PCs after truncation)
    """
    if not subject_loadings_dict:
        raise ValueError("Empty subject_loadings_dict")

    # Find minimum number of PCs across all subjects
    pc_counts = {subj: loadings.shape[0] for subj, loadings in subject_loadings_dict.items()}
    min_pcs = min(pc_counts.values())
    max_pcs = max(pc_counts.values())

    if min_pcs != max_pcs:
        print(f"\nWARNING: Subjects have varying PC counts (range: {min_pcs}-{max_pcs})")
        print(f"Truncating all subjects to {min_pcs} PCs for alignment")
        subjects_truncated = [s for s, k in pc_counts.items() if k > min_pcs]
        if subjects_truncated:
            print(f"  Truncated subjects: {', '.join(subjects_truncated[:5])}" +
                  (f" and {len(subjects_truncated)-5} more" if len(subjects_truncated) > 5 else ""))

    # Truncate all subjects to min_pcs
    truncated_loadings = {subj: loadings[:min_pcs, :]
                         for subj, loadings in subject_loadings_dict.items()}

    # Select reference
    if reference_subject is None:
        reference_subject = list(truncated_loadings.keys())[0]
        print(f"Using first subject as reference: {reference_subject}")
    elif reference_subject not in truncated_loadings:
        raise ValueError(f"Reference subject {reference_subject} not in dict")

    ref_loadings = truncated_loadings[reference_subject]
    aligned = {reference_subject: ref_loadings}
    disparities = {reference_subject: 0.0}

    print(f"\nAligning {len(truncated_loadings)} subjects to {reference_subject}")
    print(f"PC loadings shape: {ref_loadings.shape}")

    for subject, loadings in truncated_loadings.items():
        if subject == reference_subject:
            continue

        # Procrustes: find rotation R that minimizes ||ref - loadings @ R||
        # scipy.spatial.procrustes operates on rows as observations
        # So transpose: (K, D) -> (D, K) for alignment
        mtx1, mtx2, disparity = procrustes(ref_loadings.T, loadings.T)
        # mtx2 is rotated version of loadings.T, transpose back
        aligned[subject] = mtx2.T  # (K, D)
        disparities[subject] = disparity

        print(f"  {subject}: disparity = {disparity:.6f}")

    # Average aligned loadings for group
    group_loadings = np.mean(list(aligned.values()), axis=0)  # (K, D)

    print(f"\nGroup loadings computed: shape {group_loadings.shape}")
    print(f"Mean disparity: {np.mean(list(disparities.values())):.6f}")

    return {
        'aligned_loadings': aligned,
        'group_loadings': group_loadings,
        'disparities': disparities,
        'reference_subject': reference_subject,
        'n_components_used': min_pcs
    }


def _movie_feature_files():
    """
    Map feature_space names to the movie embedding CSVs in features/
    (matches the filenames the encoding models were fit with).
    """
    feature_files = {
        'alexnet_layer6': 'torchvision_alexnet_imagenet1k_v1_relu-2-16.csv',
        'sbert_layer10': 'downsampled_all-mpnet-base-v2_layer10.csv',
        'motion': 'pymoten.csv',
    }
    # feature spaces of the pooled PCA summary (Main Figure 5)
    alexnet_layers = ['maxpool2d-2-3', 'maxpool2d-2-6', 'relu-2-8', 'relu-2-10',
                      'maxpool2d-2-13', 'relu-2-16', 'relu-2-19']
    for layer, name in enumerate(alexnet_layers, start=1):
        feature_files[f'alexnet_eps0.3_layer{layer}'] = f'torchvision_alexnet_imagenet1k_v1_{name}_srp_eps0.3.csv'
    for layer in range(1, 13):
        feature_files[f'hubert_chunk120_layer{layer}'] = f'hubert_layer{layer}_chunklen120.csv'
        feature_files[f'GPT2_chunk120_layer{layer}'] = f'gpt2_layer{layer}_chunklen120_maxcontext60.csv'
        # SLIP ViT-B SimCLR: attention blocks start at index 2, mlp blocks at 6, stride 8
        feature_files[f'SimCLR_attention_eps0.3_layer{layer}'] = f'slip_vit_b_simclr_attention-3-{2 + 8*(layer-1)}_srp_eps0.3.csv'
        feature_files[f'SimCLR_embedding_eps0.3_layer{layer}'] = f'slip_vit_b_simclr_mlp-3-{6 + 8*(layer-1)}_srp_eps0.3.csv'
    return feature_files


def load_movie_embeddings(feature_space, data_dir):
    """
    Load movie embeddings for a specific feature space.

    Args:
        feature_space: e.g. 'alexnet_eps0.3_layer6', 'GPT2_chunk120_layer10', 'motion'
        data_dir: Project root directory containing features/

    Returns:
        embeddings: (1921, D) movie features aligned to TRs
    """
    feature_files = _movie_feature_files()
    if feature_space not in feature_files:
        raise ValueError(f"Unknown feature_space: {feature_space}")

    filename = feature_files[feature_space]
    filepath = os.path.join(data_dir, 'features', filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Movie embeddings not found: {filepath}")

    # Load embeddings (no header)
    embeddings = pd.read_csv(filepath, header=None).values

    print(f"Loaded {feature_space} embeddings: shape {embeddings.shape}")

    return embeddings  # (1921, D)


def project_movie_to_pcs(movie_embeddings, pc_loadings):
    """
    Project movie embeddings onto PC space.

    Args:
        movie_embeddings: (T, D) movie features
        pc_loadings: (K, D) PC loadings

    Returns:
        pc_timeseries: (T, K) PC activations over time
    """
    # Simple matrix multiplication: (T, D) @ (D, K) = (T, K)
    pc_timeseries = movie_embeddings @ pc_loadings.T

    return pc_timeseries


def save_group_loadings(group_loadings, feature_space, roi, data_dir, roi_type='func'):
    """
    Save group-level PC loadings to CSV.

    Args:
        group_loadings: (K, D) group PC loadings
        feature_space: e.g., 'alexnet_layer6'
        roi: e.g., 'MT'
        data_dir: Project root directory
        roi_type: 'func' or 'anat' (default: 'func')
    """
    output_dir = os.path.join(data_dir, 'analysis', 'WeightPCA', 'group_pca')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f'{feature_space}_{roi_type}_{roi}_group_loadings.csv'
    )

    # Convert to DataFrame: rows = PCs, columns = feature dimensions
    K, D = group_loadings.shape
    pc_names = [f'PC{i+1}' for i in range(K)]
    feature_names = [f'dim_{i}' for i in range(D)]

    df = pd.DataFrame(group_loadings, index=pc_names, columns=feature_names)
    df.index.name = 'principal_component'

    df.to_csv(output_file)
    print(f"Saved group loadings to: {output_file}")
