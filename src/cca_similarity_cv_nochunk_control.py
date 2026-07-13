"""No-chunk (TR-level) nested-CV CCA control analysis.

Modified copy of ``src/featurespace_similarity.py`` intended to sit
alongside ``src/cca_similarity_noncv_control.py``.

WHY: The non-CV control (``cca_similarity_noncv_control.py``) shows what
happens when we fit and evaluate on the same data (in-sample overfit). This
second control isolates the effect of TEMPORAL CHUNKING specifically. The
main analysis (Fig. 1C) splits train/test on 120-TR (3-min) blocks so that
adjacent, autocorrelated TRs stay together. Here we replace that
GroupKFold-on-chunks step with a plain shuffled K-fold at the individual
TR level, while keeping the outer/inner fold counts, regularization grid,
latent dimensionality, and averaging identical. Any inflation vs. Fig. 1C
is therefore attributable to information leakage from smooth
autocorrelated features across TR-level train/test splits.

>>> THIS IS A CONTROL ANALYSIS ONLY. <<<
The correlations produced here are NOT valid estimates of generalizable
shared variance and MUST NOT be reported as main-text similarity scores.
They are meant to be plotted alongside Fig. 1C values (chunked CV) and the
non-CV in-sample control to make the escalating-inflation pattern visible.
"""
import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src import encoding


# Reuse the exact 6-group definition + model feature listing helper from
# the non-CV control so the two supplementary tables are apples-to-apples.
from src.cca_similarity_noncv_control import (
    FIG1C_GROUPS, FIG1C_GROUP_ORDER, get_model_feature_list,
)


class FeatureSpaceSimilarityCVNoChunk(encoding.EncodingModel):
    """Nested-CV CCA using shuffled K-fold at the TR level (no chunking).

    Everything else — fold counts, regularization grid, top-1 canonical
    dimension, StandardScaler on train, transform on test, mean over outer
    folds — is identical to the original ``FeatureSpaceSimilarity``.
    """

    def __init__(self, args):
        self.process = 'FeatureSpaceCorrelationCVNoChunk'
        self.chunklen = args.chunklen  # not used for splits; kept for filename compat
        self.latent_dim = args.latent_dim
        self.outer_folds = args.outer_folds
        self.inner_folds = args.inner_folds
        self.random_state = args.random_state
        self.dir = args.dir
        self.out_dir = args.out_dir + '/' + self.process
        self.figure_dir = args.figure_dir + '/' + self.process
        self.apply_speech_masking = getattr(args, 'apply_speech_masking', False)
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)

        # --- features_dict (same content as the non-CV control) ---
        self.features_dict = {
            'sbert':          ['sbert_layer' + str(l) for l in range(1, 13)],
            'GPT2_1sent':     ['GPT2_1sent_layer' + str(l) for l in range(1, 13)],
            'hubert':         ['hubert_layer' + str(l) for l in range(1, 13)],
            'annotated':      ['social', 'num_agents', 'face', 'valence', 'arousal',
                               'speaking', 'turn_taking', 'mentalization',
                               'written_text', 'music'],
            'social': 'social',
            'num_agents': 'num_agents',
            'turn_taking': 'turn_taking',
            'speaking': 'speaking',
            'mentalization': 'mentalization',
            'valence': 'valence',
            'arousal': 'arousal',
            'motion': 'pymoten',
            'face': 'face',
            'indoor_outdoor': 'indoor_outdoor',
            'written_text': 'written_text',
            'pixel': 'pixel',
            'hue': 'hue',
            'amplitude': 'amplitude',
            'pitch': 'pitch',
            'music': 'music',
            'glove': 'glove',
            'word2vec': 'word2vec',
        }
        self.features_dict.update({
            'alexnet_eps0.3_layer1': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp_eps0.3',
            'alexnet_eps0.3_layer2': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp_eps0.3',
            'alexnet_eps0.3_layer3': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp_eps0.3',
            'alexnet_eps0.3_layer4': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp_eps0.3',
            'alexnet_eps0.3_layer5': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp_eps0.3',
            'alexnet_eps0.3_layer6': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-16_srp_eps0.3',
            'alexnet_eps0.3_layer7': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-19_srp_eps0.3',
        })
        self.features_dict['word2vec_chunk120'] = 'word2vec_chunklen120'
        for l in range(1, 13):
            self.features_dict[f'hubert_chunk120_layer{l}'] = f'hubert_layer{l}_chunklen120'
            self.features_dict[f'GPT2_chunk120_layer{l}']   = f'gpt2_layer{l}_chunklen120_maxcontext60'
        # SimCLR eps0.3 SRP mappings — see note in
        # ``cca_similarity_noncv_control.py``; adjust if the cluster's actual
        # SimCLR_embedding_eps0.3 filenames differ.
        for i, tracker in enumerate(range(2, 2 + 8 * 12, 8), start=1):
            self.features_dict[f'SimCLR_attention_eps0.3_layer{i}'] = \
                f'slip_vit_b_simclr_attention-3-{tracker}_srp_eps0.3'
        for i, tracker in enumerate(range(6, 6 + 8 * 12, 8), start=1):
            self.features_dict[f'SimCLR_embedding_eps0.3_layer{i}'] = \
                f'slip_vit_b_simclr_mlp-3-{tracker}_srp_eps0.3'

    def create_speech_mask(self):
        lang_feature_path = self.dir + '/features/word2vec.csv'
        lang_data = np.array(pd.read_csv(lang_feature_path, header=None)).astype('float32')
        return ~np.all(lang_data == 0, axis=1)

    def load_feature(self, feature_name):
        fname = self.features_dict[feature_name]
        filepath = self.dir + '/features/' + fname.lower() + '.csv'
        data = np.array(pd.read_csv(filepath, header=None))
        if self.apply_speech_masking:
            data = data[self.create_speech_mask()]
        return data.astype('float32')

    # -------------------------------------------------- no-chunk nested CV
    def canonical_correlation_analysis_nochunk(self, feature_names):
        """Nested-CV rCCA using shuffled K-fold at the TR level.

        Structurally identical to the ``regularized=True`` branch of the
        original ``FeatureSpaceSimilarity.canonical_correlation_analysis``,
        with the sole difference that both outer and inner splitters are
        ``KFold(shuffle=True)`` on TR indices rather than
        ``GroupKFold`` on 120-TR chunk indices.
        """
        from cca_zoo.model_selection import GridSearchCV
        from cca_zoo.linear import rCCA

        X1 = self.load_feature(feature_names[0])
        X2 = self.load_feature(feature_names[1])
        n_samples = min(X1.shape[0], X2.shape[0])
        X1, X2 = X1[:n_samples], X2[:n_samples]

        reg_params = np.logspace(-5, 0, 5)
        param_grid = {"c": [reg_params, reg_params]}

        cv_outer = KFold(n_splits=self.outer_folds, shuffle=True,
                         random_state=self.random_state)
        correlations_test = []
        for i, (train_outer, test_outer) in enumerate(
                cv_outer.split(np.arange(n_samples))):
            scaler_X, scaler_Y = StandardScaler(), StandardScaler()
            train1 = np.nan_to_num(scaler_X.fit_transform(X1[train_outer])).astype('float32')
            train2 = np.nan_to_num(scaler_Y.fit_transform(X2[train_outer])).astype('float32')
            test1 = np.nan_to_num(scaler_X.transform(X1[test_outer])).astype('float32')
            test2 = np.nan_to_num(scaler_Y.transform(X2[test_outer])).astype('float32')

            cv_inner = KFold(n_splits=self.inner_folds, shuffle=True,
                             random_state=self.random_state)
            model = GridSearchCV(
                rCCA(latent_dimensions=self.latent_dim),
                param_grid=param_grid,
                cv=cv_inner, verbose=1, error_score='raise',
            ).fit((train1, train2))

            fold_corr = model.best_estimator_.average_pairwise_correlations(
                (test1, test2)).mean()
            correlations_test.append(float(fold_corr))
            print(f'  outer fold {i}: {fold_corr:.4f}', flush=True)

        return {
            'correlation_top1_meanfold': float(np.mean(correlations_test)),
            'correlation_top1_perfold': correlations_test,
            'n_dims_1': int(X1.shape[1]),
            'n_dims_2': int(X2.shape[1]),
            'n_samples': int(n_samples),
        }

    def run_pair(self, feature_names):
        result = self.canonical_correlation_analysis_nochunk(feature_names)
        f1, f2 = feature_names
        label = (f'{f1}-{f2}_latent_dim-{self.latent_dim}_'
                 f'outer-{self.outer_folds}_inner-{self.inner_folds}')
        if self.apply_speech_masking:
            label += '_speech-masked'
        filepath = os.path.join(self.out_dir, label + '.csv')
        # per-pair CSV mirrors the original pipeline (one scalar per file) so
        # the same compile step can consume it.
        pd.DataFrame([{
            'cca_cv_nochunk': result['correlation_top1_meanfold'],
            'per_fold': str(result['correlation_top1_perfold']),
            'n_dims_1': result['n_dims_1'],
            'n_dims_2': result['n_dims_2'],
            'n_samples': result['n_samples'],
        }]).to_csv(filepath, index=False)
        print(f'saved {filepath}', flush=True)
        return result


# ============================================================================
# Compile step: aggregate per-pair no-chunk CV CSVs into the extended
# comparison table (Fig 1C CV + no-chunk CV + non-CV) and produce plots.
# ============================================================================
def _read_nochunk_csv(path):
    try:
        df = pd.read_csv(path)
        return {
            'cca_cv_nochunk': float(df.iloc[0]['cca_cv_nochunk']),
            'n_dims_1': int(df.iloc[0].get('n_dims_1', np.nan)),
            'n_dims_2': int(df.iloc[0].get('n_dims_2', np.nan)),
            'n_samples': int(df.iloc[0].get('n_samples', np.nan)),
        }
    except Exception:
        return {'cca_cv_nochunk': np.nan}


def _find_nochunk_result(model_out_dir, f1, f2, latent_dim,
                         outer_folds, inner_folds, speech_masked):
    """Match a per-pair no-chunk CSV, tolerating (f1,f2) order + case."""
    suffix = f'_latent_dim-{latent_dim}_outer-{outer_folds}_inner-{inner_folds}'
    if speech_masked:
        suffix += '_speech-masked'
    for a, b in [(f1, f2), (f2, f1), (f1.lower(), f2.lower()), (f2.lower(), f1.lower())]:
        p = os.path.join(model_out_dir, f'{a}-{b}{suffix}.csv')
        if os.path.exists(p):
            return _read_nochunk_csv(p)
    return {'cca_cv_nochunk': np.nan}


def _existing_cv_value(analysis_dir, f1, f2, latent_dim, speech_masked):
    """Reused from the non-CV control — matches existing Fig 1C CSVs."""
    for a, b in [(f1, f2), (f2, f1), (f1.lower(), f2.lower()), (f2.lower(), f1.lower())]:
        base = f'{a}-{b}_latent_dim-{latent_dim}'
        if speech_masked:
            base += '_speech-masked'
        p = os.path.join(analysis_dir, base + '.csv')
        if os.path.exists(p):
            try:
                return float(pd.read_csv(p, header=None).iloc[0, 0])
            except Exception:
                pass
    return np.nan


def compile_all_three_for_model(model, model_name, cv_chunked_dir,
                                noncv_per_pair_csv):
    """Per-model three-way compile (Fig 1C CV + no-chunk CV + non-CV) over
    every within-model feature pair. Mirrors ``compile_all_three`` but
    iterates the flat feature list from ``get_model_feature_list``."""
    features = get_model_feature_list(model_name)
    noncv_df = pd.read_csv(noncv_per_pair_csv) if os.path.exists(noncv_per_pair_csv) \
        else pd.DataFrame(columns=['feature_1', 'feature_2', 'cca_noncv'])

    per_pair_rows = []
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if j <= i:
                continue
            cv_chunk = _existing_cv_value(
                cv_chunked_dir, f1, f2,
                model.latent_dim, model.apply_speech_masking)
            nochunk = _find_nochunk_result(
                model.out_dir, f1, f2, model.latent_dim,
                model.outer_folds, model.inner_folds,
                model.apply_speech_masking)
            noncv_row = noncv_df[
                ((noncv_df['feature_1'] == f1) & (noncv_df['feature_2'] == f2)) |
                ((noncv_df['feature_1'] == f2) & (noncv_df['feature_2'] == f1))]
            noncv_val = float(noncv_row['cca_noncv'].iloc[0]) if len(noncv_row) else np.nan
            per_pair_rows.append({
                'model': model_name,
                'feature_1': f1, 'feature_2': f2,
                'cca_cv_chunked': cv_chunk,
                'cca_cv_nochunk': nochunk['cca_cv_nochunk'],
                'cca_noncv': noncv_val,
                'n_dims_1': nochunk.get('n_dims_1', np.nan),
                'n_dims_2': nochunk.get('n_dims_2', np.nan),
                'n_samples': nochunk.get('n_samples', np.nan),
            })
    per_pair_df = pd.DataFrame(per_pair_rows)
    out_dir = os.path.join(model.out_dir, model_name)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    per_pair_path = os.path.join(out_dir, 'per_feature_pair_three_way.csv')
    per_pair_df.to_csv(per_pair_path, index=False)

    fig_dir = os.path.join(model.figure_dir, model_name)
    Path(fig_dir).mkdir(exist_ok=True, parents=True)
    plot_path = os.path.join(fig_dir,
                             f'{model_name}_cv_chunk_vs_nochunk_vs_noncv.png')
    _plot_three_way_bars_flat(per_pair_df, plot_path, model_name)

    print('Saved:')
    print(' ', per_pair_path)
    print(' ', plot_path)


def _plot_three_way_bars_flat(per_pair_df, output_path, model_name):
    """Scatter version of the three-way bar chart — one point per pair,
    stacked by CV condition. Better than bars when there are hundreds of
    pairs (as in a full within-model comparison)."""
    import matplotlib.pyplot as plt
    if per_pair_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(per_pair_df))
    ax.scatter(x, per_pair_df['cca_cv_chunked'], s=8, alpha=0.7,
               label='Fig. 1C (chunked CV, 120-TR)', color='#2b8a3e')
    ax.scatter(x, per_pair_df['cca_cv_nochunk'], s=8, alpha=0.7,
               label='CV, no-chunk (control)', color='#f08c00')
    ax.scatter(x, per_pair_df['cca_noncv'], s=8, alpha=0.7,
               label='non-CV in-sample (control)', color='#c92a2a')
    ax.set_ylabel('top-1 canonical correlation')
    ax.set_xlabel('feature-pair index (within-model)')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{model_name}: chunked CV vs no-chunk CV vs in-sample\n'
                 '[middle & right series are OVERFIT-PRONE controls]')
    ax.legend(loc='center right', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def compile_all_three(model, cv_chunked_dir, noncv_per_pair_csv):
    """Build the extended comparison table with three CV/non-CV variants.

    Reads:
      * per-pair no-chunk CV CSVs from ``model.out_dir``
      * per-pair chunked CV values from ``cv_chunked_dir`` (existing Fig 1C
        outputs, one scalar per CSV)
      * per-layer non-CV control table from ``noncv_per_pair_csv``
    Writes group-level and per-layer comparison tables and a bar chart.
    """
    # --- non-CV values (per-layer pairs) ---
    noncv_df = pd.read_csv(noncv_per_pair_csv) if os.path.exists(noncv_per_pair_csv) \
        else pd.DataFrame(columns=['feature_1', 'feature_2', 'cca_noncv',
                                   'n_dims_1', 'n_dims_2', 'n_samples',
                                   'group_1', 'group_2'])

    per_pair_rows = []
    for i, g1 in enumerate(FIG1C_GROUP_ORDER):
        for j, g2 in enumerate(FIG1C_GROUP_ORDER):
            if j <= i:
                continue
            for f1 in FIG1C_GROUPS[g1]:
                for f2 in FIG1C_GROUPS[g2]:
                    if f1 == f2:
                        continue
                    cv_chunk = _existing_cv_value(
                        cv_chunked_dir, f1, f2,
                        model.latent_dim, model.apply_speech_masking)
                    nochunk = _find_nochunk_result(
                        model.out_dir, f1, f2, model.latent_dim,
                        model.outer_folds, model.inner_folds,
                        model.apply_speech_masking)
                    # non-CV lookup by (feature_1, feature_2) order tolerant
                    noncv_row = noncv_df[
                        ((noncv_df['feature_1'] == f1) & (noncv_df['feature_2'] == f2)) |
                        ((noncv_df['feature_1'] == f2) & (noncv_df['feature_2'] == f1))]
                    noncv_val = float(noncv_row['cca_noncv'].iloc[0]) \
                        if len(noncv_row) else np.nan
                    per_pair_rows.append({
                        'group_1': g1, 'group_2': g2,
                        'feature_1': f1, 'feature_2': f2,
                        'cca_cv_chunked': cv_chunk,
                        'cca_cv_nochunk': nochunk['cca_cv_nochunk'],
                        'cca_noncv': noncv_val,
                        'n_dims_1': nochunk.get('n_dims_1', np.nan),
                        'n_dims_2': nochunk.get('n_dims_2', np.nan),
                        'n_samples': nochunk.get('n_samples', np.nan),
                    })
    per_pair_df = pd.DataFrame(per_pair_rows)

    group_df = (per_pair_df
                .groupby(['group_1', 'group_2'], as_index=False)
                .agg(cca_cv_chunked=('cca_cv_chunked', 'mean'),
                     cca_cv_nochunk=('cca_cv_nochunk', 'mean'),
                     cca_noncv=('cca_noncv', 'mean'),
                     n_dims_1=('n_dims_1', 'mean'),
                     n_dims_2=('n_dims_2', 'mean'),
                     n_samples=('n_samples', 'mean'),
                     n_layer_pairs=('feature_1', 'count')))

    per_pair_path = os.path.join(model.out_dir, 'per_layer_pair_three_way.csv')
    group_path    = os.path.join(model.out_dir, 'group_pair_three_way.csv')
    per_pair_df.to_csv(per_pair_path, index=False)
    group_df.to_csv(group_path, index=False)
    print('Saved:')
    print(' ', per_pair_path)
    print(' ', group_path)

    # ----- bar chart per group pair -----
    plot_path = os.path.join(model.figure_dir, 'cca_cv_chunk_vs_nochunk_vs_noncv.png')
    _plot_three_way_bars(group_df, plot_path)
    print(' ', plot_path)


def _plot_three_way_bars(group_df, output_path):
    import matplotlib.pyplot as plt
    valid = group_df.dropna(subset=['cca_cv_chunked', 'cca_cv_nochunk', 'cca_noncv'],
                            how='all').copy()
    if valid.empty:
        print('No rows to plot; skipping bar chart.')
        return
    valid['label'] = valid['group_1'] + '\nx\n' + valid['group_2']
    x = np.arange(len(valid))
    w = 0.28
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(valid)), 4.5))
    ax.bar(x - w, valid['cca_cv_chunked'], w,
           label='Fig. 1C  (chunked CV, 120-TR)', color='#2b8a3e')
    ax.bar(x,     valid['cca_cv_nochunk'], w,
           label='CV, no-chunk (control)',        color='#f08c00')
    ax.bar(x + w, valid['cca_noncv'],     w,
           label='non-CV in-sample (control)',    color='#c92a2a')
    ax.set_xticks(x)
    ax.set_xticklabels(valid['label'], fontsize=8)
    ax.set_ylabel('top-1 canonical correlation')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(('Effect of temporal chunking and cross-validation on CCA '
                  'similarity\n[middle & right bars are OVERFIT-PRONE '
                  'controls — not real similarity estimates]'), fontsize=10)
    ax.legend(loc='upper left', fontsize=8, ncol=1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=('CCA with nested CV but TR-level shuffled K-fold '
                     '(no 120-TR chunking). CONTROL ANALYSIS ONLY: exposes '
                     'temporal-autocorrelation leakage across train/test.'))
    parser.add_argument('--mode',
                        choices=['pair', 'compile', 'compile-model'],
                        default='pair',
                        help=('"pair" = run one feature pair via the cluster '
                              'sbatch pipeline; "compile" = aggregate per-pair '
                              'CSVs + Fig 1C CV values + non-CV values into '
                              'the Fig-1C 6-group three-way comparison table + '
                              'bar chart; "compile-model" = same but for every '
                              'within-model pair of the encoding model named '
                              'by --model.'))
    parser.add_argument('--model', type=str, default=None,
                        help=('Encoding-model name for --mode compile-model. '
                              'E.g. joint_chunk120, '
                              'vislang_simclr_gpt2_chunk120_eps0.3.'))
    parser.add_argument('--features', type=str, default='',
                        help='For --mode pair, e.g. "motion-social".')
    parser.add_argument('--chunklen', type=int, default=120)
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--apply-speech-masking', action='store_true')
    parser.add_argument('--dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/ubiquitous-vis')
    parser.add_argument('--out_dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/ubiquitous-vis/analysis')
    parser.add_argument('--figure_dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/ubiquitous-vis/figures')
    parser.add_argument('--cv-chunked-dir', type=str, default=None,
                        help=('Directory of existing Fig 1C per-pair CV CSVs. '
                              'Defaults to <dir>/analysis/FeatureSpaceCorrelation.'))
    parser.add_argument('--noncv-per-pair-csv', type=str, default=None,
                        help=('Path to the per-layer non-CV control table '
                              '(per_layer_pair_cv_vs_noncv.csv). Defaults to '
                              '<out_dir>/FeatureSpaceCorrelationNonCV/'
                              'per_layer_pair_cv_vs_noncv.csv.'))
    args = parser.parse_args()

    model = FeatureSpaceSimilarityCVNoChunk(args)

    if args.mode == 'pair':
        assert '-' in args.features, '--features must look like "f1-f2"'
        f1, f2 = args.features.split('-', 1)
        print(f'== no-chunk CV, {f1} vs {f2} ==', flush=True)
        r = model.run_pair([f1, f2])
        print('mean top-1 CCA (across outer folds):',
              r['correlation_top1_meanfold'])
        return

    # compile
    cv_chunked_dir = args.cv_chunked_dir or os.path.join(
        args.dir, 'analysis', 'FeatureSpaceCorrelation')
    if args.mode == 'compile-model':
        assert args.model, '--mode compile-model requires --model MODEL_NAME'
        noncv_per_pair_csv = args.noncv_per_pair_csv or os.path.join(
            args.out_dir, 'FeatureSpaceCorrelationNonCV',
            args.model, 'per_feature_pair_cv_vs_noncv.csv')
        compile_all_three_for_model(model, args.model, cv_chunked_dir,
                                    noncv_per_pair_csv)
        return

    noncv_per_pair_csv = args.noncv_per_pair_csv or os.path.join(
        args.out_dir, 'FeatureSpaceCorrelationNonCV',
        'per_layer_pair_cv_vs_noncv.csv')
    compile_all_three(model, cv_chunked_dir, noncv_per_pair_csv)


if __name__ == '__main__':
    main()
