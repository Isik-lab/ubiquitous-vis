"""Non-CV (in-sample) CCA control analysis.

Modified copy of src/featurespace_similarity.py.

WHY: A reviewer asked why the cross-validated CCA correlations in Fig. 1C are
low (e.g. image vs. motion = 0.02) when high-dimensional feature spaces
"should" be able to find some linear combination that correlates with the
other view. Fig. 1C uses nested cross-validation precisely to prevent that
kind of in-sample overfit inflation. This script reruns the same CCA on the
same feature spaces WITHOUT cross-validation (fit = test, minimal L2
regularization) to demonstrate that in-sample correlations are trivially
close to 1.0, so the low CV numbers in Fig. 1C reflect proper correction for
overfitting rather than an error or bug.

>>> THIS IS AN OVERFIT-PRONE CONTROL ANALYSIS. <<<
The correlations produced here are NOT valid estimates of generalizable
shared variance and MUST NOT be reported in the main text as similarity
scores. They are only meant for the reviewer response / supplementary
figure showing that removing CV inflates the correlation.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import encoding


# ---------------------------------------------------------------------------
# Six "groups" used in Fig. 1C: image, motion, social interaction, speech,
# word, multi-sentence. Each group is a list of per-layer feature names that
# match the CSV filenames used by the existing pipeline.
# ---------------------------------------------------------------------------
FIG1C_GROUPS = {
    'image':           [f'alexnet_eps0.3_layer{i}' for i in range(1, 8)],
    'motion':          ['motion'],
    'social':          ['social'],
    'speech':          [f'hubert_chunk120_layer{i}' for i in range(1, 13)],
    'word':            ['word2vec_chunk120'],
    'multi_sentence':  [f'GPT2_chunk120_layer{i}' for i in range(1, 13)],
}
FIG1C_GROUP_ORDER = ['image', 'motion', 'social', 'speech', 'word', 'multi_sentence']


# ---------------------------------------------------------------------------
# Feature listings for any encoding-model name from helpers.get_models_dict().
# Used by the --model CLI to enumerate all within-model feature-space pairs.
# ---------------------------------------------------------------------------
def get_model_feature_list(model_name):
    """Return the flat list of feature names in an encoding model,
    resolved from ``helpers.get_models_dict()``. Order is preserved so
    the resulting NxN comparison matrix has a sensible feature ordering.
    Deduplication is applied while preserving first-seen order."""
    from src import helpers
    models_dict = helpers.get_models_dict()
    if model_name not in models_dict:
        raise KeyError(f'unknown model {model_name!r}; keys include '
                       f'joint_chunk120, vislang_simclr_gpt2_chunk120_eps0.3')
    seen, out = set(), []
    for f in models_dict[model_name]:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


class FeatureSpaceSimilarityNonCV(encoding.EncodingModel):
    """In-sample CCA control (no cross-validation, minimal regularization).

    All feature loading + preprocessing is identical to
    ``FeatureSpaceSimilarity`` — the only difference is the fitting procedure.
    """

    def __init__(self, args):
        self.process = 'FeatureSpaceCorrelationNonCV'
        self.chunklen = args.chunklen
        self.latent_dim = args.latent_dim
        self.reg_param = args.reg_param
        self.apply_windowing = not args.skip_windowing
        self.dir = args.dir
        self.out_dir = args.out_dir + '/' + self.process
        self.figure_dir = args.figure_dir + '/' + self.process
        self.apply_speech_masking = getattr(args, 'apply_speech_masking', False)
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)

        # --- features_dict copied from featurespace_similarity.py + chunk120
        # entries needed for Fig. 1C. Kept intentionally verbose to minimize
        # divergence from the original pipeline.
        self.features_dict = {
            'sbert':          ['sbert_layer' + str(l) for l in range(1, 13)],
            'GPT2_1sent':     ['GPT2_1sent_layer' + str(l) for l in range(1, 13)],
            'GPT2_3sent':     ['GPT2_3sent_layer' + str(l) for l in range(1, 13)],
            'GPT2_1word':     ['GPT2_1word_layer' + str(l) for l in range(1, 13)],
            'GPT2_4s':        ['GPT2_4s_layer' + str(l) for l in range(1, 13)],
            'GPT2_8s':        ['GPT2_8s_layer' + str(l) for l in range(1, 13)],
            'GPT2_16s':       ['GPT2_16s_layer' + str(l) for l in range(1, 13)],
            'GPT2_24s':       ['GPT2_24s_layer' + str(l) for l in range(1, 13)],
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
        # AlexNet SRP (eps0.3) layers used by Fig. 1C. Filenames verified in
        # the local features/ directory; case is irrelevant since the loader
        # lowercases the filename.
        alexnet_eps03 = {
            'alexnet_eps0.3_layer1': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp_eps0.3',
            'alexnet_eps0.3_layer2': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp_eps0.3',
            'alexnet_eps0.3_layer3': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp_eps0.3',
            'alexnet_eps0.3_layer4': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp_eps0.3',
            'alexnet_eps0.3_layer5': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp_eps0.3',
            'alexnet_eps0.3_layer6': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-16_srp_eps0.3',
            'alexnet_eps0.3_layer7': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-19_srp_eps0.3',
        }
        self.features_dict.update(alexnet_eps03)
        # chunklen-120 language + speech mappings (Fig. 1C uses these).
        self.features_dict['word2vec_chunk120'] = 'word2vec_chunklen120'
        for l in range(1, 13):
            self.features_dict[f'hubert_chunk120_layer{l}'] = f'hubert_layer{l}_chunklen120'
            self.features_dict[f'GPT2_chunk120_layer{l}']   = f'gpt2_layer{l}_chunklen120_maxcontext60'
        # base alexnet (non-eps) mappings from featurespace_similarity.py
        self.features_dict.update({
            'alexnet_layer1': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp',
            'alexnet_layer2': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp',
            'alexnet_layer3': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp',
            'alexnet_layer4': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp',
            'alexnet_layer5': 'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp',
            'alexnet_layer6': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-16',
            'alexnet_layer7': 'torchvision_alexnet_imagenet1k_v1_ReLU-2-19',
        })
        # SimCLR eps0.3 SRP mappings (attention + mlp/embedding). Pattern
        # verified for attention (matches encoding_speech_eval.py). Embedding
        # mapping follows the same tracker convention as the base
        # SimCLR_embedding entry in encoding.py — filenames on the cluster
        # should match ``slip_vit_b_simclr_mlp-3-{tracker}_srp_eps0.3``. If
        # the actual cluster filename differs, adjust here.
        for i, tracker in enumerate(range(2, 2 + 8 * 12, 8), start=1):
            self.features_dict[f'SimCLR_attention_eps0.3_layer{i}'] = \
                f'slip_vit_b_simclr_attention-3-{tracker}_srp_eps0.3'
        for i, tracker in enumerate(range(6, 6 + 8 * 12, 8), start=1):
            self.features_dict[f'SimCLR_embedding_eps0.3_layer{i}'] = \
                f'slip_vit_b_simclr_mlp-3-{tracker}_srp_eps0.3'

    # ------------------------------------------------------------------ helpers
    def create_speech_mask(self):
        """Boolean mask of speech periods (True) — copied from
        FeatureSpaceSimilarity so preprocessing paths stay identical."""
        lang_feature_path = self.dir + '/features/word2vec.csv'
        lang_data = np.array(pd.read_csv(lang_feature_path, header=None)).astype('float32')
        return ~np.all(lang_data == 0, axis=1)

    def load_feature(self, feature_name):
        """Load one feature CSV. Returns (n_samples, n_features) float32."""
        fname = self.features_dict[feature_name]
        filepath = self.dir + '/features/' + fname.lower() + '.csv'
        data = np.array(pd.read_csv(filepath, header=None))
        if self.apply_speech_masking:
            data = data[self.create_speech_mask()]
        return data.astype('float32')

    # -------------------------------------------------------- non-CV CCA core
    def canonical_correlation_analysis_noncv(self, feature_names):
        """Fit CCA on the FULL dataset (no train/test split) and compute the
        top-``latent_dim`` canonical correlations on the same data.

        Uses ``cca_zoo.linear.rCCA`` with minimal L2 regularization
        (``self.reg_param``, default 1e-5) for numerical stability when the
        feature dimensionality exceeds the sample count — which is the
        condition the reviewer's concern is about.
        """
        from cca_zoo.linear import rCCA

        X1 = self.load_feature(feature_names[0])
        X2 = self.load_feature(feature_names[1])
        n_samples = min(X1.shape[0], X2.shape[0])
        X1, X2 = X1[:n_samples], X2[:n_samples]

        if self.apply_windowing:
            n_chunks = n_samples // self.chunklen
            keep = n_chunks * self.chunklen
            X1, X2 = X1[:keep], X2[:keep]

        scaler_X, scaler_Y = StandardScaler(), StandardScaler()
        X1s = np.nan_to_num(scaler_X.fit_transform(X1)).astype('float32')
        X2s = np.nan_to_num(scaler_Y.fit_transform(X2)).astype('float32')

        model = rCCA(latent_dimensions=self.latent_dim,
                     c=[self.reg_param, self.reg_param])
        model.fit((X1s, X2s))

        # top-1 canonical correlation on the same data used for fitting
        corrs = np.atleast_1d(model.average_pairwise_correlations((X1s, X2s)))
        return {
            'correlation_top1': float(corrs[0]),
            'n_dims_1': int(X1s.shape[1]),
            'n_dims_2': int(X2s.shape[1]),
            'n_samples': int(X1s.shape[0]),
        }

    # ----------------------------------------------------- single-pair driver
    def run_pair(self, feature_names):
        result = self.canonical_correlation_analysis_noncv(feature_names)
        f1, f2 = feature_names
        label = f'{f1}-{f2}_latent_dim-{self.latent_dim}_reg-{self.reg_param}'
        if not self.apply_windowing:
            label += '_no-window'
        if self.apply_speech_masking:
            label += '_speech-masked'
        filepath = os.path.join(self.out_dir, label + '.csv')
        pd.DataFrame([result]).to_csv(filepath, index=False)
        return result


# ============================================================================
# Group-level driver: enumerates all 6-group pairs from Fig 1C, aggregates
# per-layer non-CV correlations, and produces a comparison table + heatmap.
# ============================================================================
def _existing_cv_value(analysis_dir, f1, f2, latent_dim, speech_masked):
    """Look up an existing per-layer CV result from
    analysis/FeatureSpaceCorrelation. Tries both (f1,f2) orderings and both
    common case variants (macOS default APFS is case-insensitive but Linux is
    not, so we try lowercase too)."""
    base_variants = [(f1, f2), (f2, f1),
                     (f1.lower(), f2.lower()), (f2.lower(), f1.lower())]
    for a, b in base_variants:
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


def run_all_pairs_for_model(model: FeatureSpaceSimilarityNonCV,
                            model_name: str,
                            cv_analysis_dir: str,
                            skip_missing_features: bool = False):
    """Run non-CV CCA for every (feature_i, feature_j) pair within an
    encoding model's flat feature list (upper triangle only). One row per
    pair. Loads existing chunked-CV values from ``cv_analysis_dir`` for
    side-by-side comparison. If ``skip_missing_features`` is True, pairs
    that reference a feature whose CSV is not on disk are skipped with a
    warning rather than raising."""
    features = get_model_feature_list(model_name)
    per_pair_rows = []
    missing = set()
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if j <= i:
                continue
            print(f'[{model_name}] ({i+1}/{len(features)}) {f1}  vs  {f2}',
                  flush=True)
            if skip_missing_features:
                ok = True
                for f in (f1, f2):
                    if f not in model.features_dict:
                        missing.add(f); ok = False; continue
                    fname = model.features_dict[f]
                    if not os.path.exists(
                            model.dir + '/features/' + fname.lower() + '.csv'):
                        missing.add(f); ok = False
                if not ok:
                    per_pair_rows.append({
                        'model': model_name, 'feature_1': f1, 'feature_2': f2,
                        'cca_noncv': np.nan, 'cca_cv': np.nan,
                        'n_dims_1': np.nan, 'n_dims_2': np.nan,
                        'n_samples': np.nan,
                    })
                    continue
            r = model.canonical_correlation_analysis_noncv([f1, f2])
            cv = _existing_cv_value(cv_analysis_dir, f1, f2,
                                    model.latent_dim, model.apply_speech_masking)
            per_pair_rows.append({
                'model': model_name, 'feature_1': f1, 'feature_2': f2,
                'cca_noncv': r['correlation_top1'],
                'cca_cv': cv,
                'n_dims_1': r['n_dims_1'],
                'n_dims_2': r['n_dims_2'],
                'n_samples': r['n_samples'],
            })
    if missing:
        print(f'[{model_name}] SKIPPED features (no CSV on disk): '
              f'{sorted(missing)}', flush=True)
    per_pair_df = pd.DataFrame(per_pair_rows)

    n = len(features)
    matrix_cv = np.full((n, n), np.nan)
    matrix_noncv = np.full((n, n), np.nan)
    for _, row in per_pair_df.iterrows():
        i = features.index(row['feature_1'])
        j = features.index(row['feature_2'])
        matrix_cv[i, j] = matrix_cv[j, i] = row['cca_cv']
        matrix_noncv[i, j] = matrix_noncv[j, i] = row['cca_noncv']
    return per_pair_df, matrix_cv, matrix_noncv, features


def run_all_fig1c_groups(model: FeatureSpaceSimilarityNonCV,
                         cv_analysis_dir: str):
    """Run non-CV CCA for every (group_i, group_j) sub-layer pair, aggregate
    into the 6x6 matrix, and pair each cell with the corresponding CV value.

    Returns
    -------
    per_pair_df : long-form DataFrame with one row per layer-layer sub-pair
                  (feature_1, feature_2, cca_cv, cca_noncv, n_dims_1,
                  n_dims_2, n_samples, group_1, group_2).
    group_df    : long-form DataFrame with one row per group-group pair
                  (cca_cv/cca_noncv averaged over sub-pairs).
    matrix_cv, matrix_noncv : 6x6 numpy arrays indexed by FIG1C_GROUP_ORDER.
    """
    groups = FIG1C_GROUPS
    order = FIG1C_GROUP_ORDER

    per_pair_rows = []
    for i, g1 in enumerate(order):
        for j, g2 in enumerate(order):
            if j <= i:
                # only strict upper triangle; symmetric matrix so lower is
                # filled from the same values; diagonals are not shown in
                # the Fig-1C-style lower-triangle heatmap.
                continue
            for f1 in groups[g1]:
                for f2 in groups[g2]:
                    if f1 == f2:
                        continue
                    print(f'[{g1} x {g2}]  {f1}  vs  {f2}', flush=True)
                    r = model.canonical_correlation_analysis_noncv([f1, f2])
                    cv = _existing_cv_value(
                        cv_analysis_dir, f1, f2,
                        model.latent_dim, model.apply_speech_masking)
                    per_pair_rows.append({
                        'group_1': g1, 'group_2': g2,
                        'feature_1': f1, 'feature_2': f2,
                        'cca_noncv': r['correlation_top1'],
                        'cca_cv': cv,
                        'n_dims_1': r['n_dims_1'],
                        'n_dims_2': r['n_dims_2'],
                        'n_samples': r['n_samples'],
                    })

    per_pair_df = pd.DataFrame(per_pair_rows)

    # aggregate to 6x6 groups (mean over layer-layer sub-pairs)
    agg = (per_pair_df
           .groupby(['group_1', 'group_2'], as_index=False)
           .agg(cca_cv=('cca_cv', 'mean'),
                cca_noncv=('cca_noncv', 'mean'),
                n_dims_1=('n_dims_1', 'mean'),
                n_dims_2=('n_dims_2', 'mean'),
                n_samples=('n_samples', 'mean'),
                n_layer_pairs=('feature_1', 'count')))

    n = len(order)
    matrix_cv = np.full((n, n), np.nan)
    matrix_noncv = np.full((n, n), np.nan)
    for _, row in agg.iterrows():
        i = order.index(row['group_1'])
        j = order.index(row['group_2'])
        matrix_cv[i, j] = matrix_cv[j, i] = row['cca_cv']
        matrix_noncv[i, j] = matrix_noncv[j, i] = row['cca_noncv']

    return per_pair_df, agg, matrix_cv, matrix_noncv


def plot_cv_vs_noncv_heatmaps(matrix_cv, matrix_noncv, group_labels,
                              output_path, subtitle=None):
    """Side-by-side lower-triangle heatmaps of CV vs non-CV correlations."""
    import matplotlib.pyplot as plt
    from src import plotting_helpers
    try:
        cmap = plotting_helpers.get_cmaps()['matrix_green']
    except Exception:
        cmap = 'Greens'

    labels = [g.replace('_', '\n') for g in group_labels]
    n = len(labels)

    def _prep(m):
        m = m.copy()
        m[np.triu_indices(n, 0)] = np.nan
        return m

    fig, axes = plt.subplots(1, 2, figsize=(2 * (n * 1.4), n * 1.4))
    for ax, m, title in zip(
            axes,
            [_prep(matrix_cv), _prep(matrix_noncv)],
            [f'Cross-validated (Fig. 1C)',
             f'In-sample (non-CV control)\n[OVERFIT-PRONE — not a real estimate]']):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=10)
        for i in range(n):
            for j in range(n):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center',
                            fontsize=9, color='black')
        fig.colorbar(im, ax=ax, shrink=0.7, label='top-1 canonical correlation')

    if subtitle:
        fig.suptitle(subtitle, fontsize=10, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=('Non-CV (in-sample) CCA control analysis. '
                     'OVERFIT-PRONE by design — for reviewer response only.'))
    parser.add_argument('--mode', choices=['pair', 'all-fig1c', 'all-model'],
                        default='all-fig1c',
                        help=('"pair" = single feature-pair (like the '
                              'original featurespace_similarity.py CLI); '
                              '"all-fig1c" = run every layer-layer sub-pair '
                              'across the 6 Fig. 1C groups and produce the '
                              '6x6 comparison table + heatmap; "all-model" = '
                              'run every within-model feature pair for the '
                              'encoding model named by --model.'))
    parser.add_argument('--model', type=str, default=None,
                        help=('Encoding-model name (key of '
                              'helpers.get_models_dict()) for --mode all-model. '
                              'E.g. joint_chunk120, '
                              'vislang_simclr_gpt2_chunk120_eps0.3.'))
    parser.add_argument('--skip-missing-features', action='store_true',
                        help=('For --mode all-model, skip pairs whose feature '
                              'CSV is missing on disk (useful for local runs '
                              'where some cluster-only features are absent).'))
    parser.add_argument('--features', type=str, default='',
                        help='Only for --mode pair. e.g. "motion-social".')
    parser.add_argument('--chunklen', type=int, default=120)
    parser.add_argument('--latent_dim', type=int, default=1)
    parser.add_argument('--reg-param', type=float, default=1e-5,
                        help=('L2 regularization on both views (rCCA c=). '
                              'Default 1e-5 = smallest value from the '
                              'existing CV grid (np.logspace(-5,0,5)). Use '
                              '0 for pure CCA if numerically stable.'))
    parser.add_argument('--skip-windowing', action='store_true',
                        help='Skip the trim-to-chunklen truncation.')
    parser.add_argument('--apply-speech-masking', action='store_true')
    parser.add_argument('--dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/ubiquitous-vis')
    parser.add_argument('--out_dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/ubiquitous-vis/analysis')
    parser.add_argument('--figure_dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/ubiquitous-vis/figures')
    parser.add_argument('--cv-analysis-dir', type=str, default=None,
                        help=('Directory of existing per-pair CV CSVs from '
                              'the original pipeline. Defaults to '
                              '<dir>/analysis/FeatureSpaceCorrelation.'))
    args = parser.parse_args()

    model = FeatureSpaceSimilarityNonCV(args)

    if args.mode == 'pair':
        assert '-' in args.features, '--features must look like "f1-f2"'
        f1, f2 = args.features.split('-', 1)
        r = model.run_pair([f1, f2])
        print(r)
        return

    if args.mode == 'all-model':
        assert args.model, '--mode all-model requires --model MODEL_NAME'
        cv_dir = args.cv_analysis_dir or os.path.join(
            args.dir, 'analysis', 'FeatureSpaceCorrelation')
        per_pair_df, matrix_cv, matrix_noncv, features = \
            run_all_pairs_for_model(model, args.model, cv_dir,
                                    skip_missing_features=args.skip_missing_features)

        out_dir = os.path.join(model.out_dir, args.model)
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        per_pair_path = os.path.join(out_dir, 'per_feature_pair_cv_vs_noncv.csv')
        matrix_cv_path = os.path.join(out_dir, 'feature_matrix_cv.csv')
        matrix_noncv_path = os.path.join(out_dir, 'feature_matrix_noncv.csv')
        per_pair_df.to_csv(per_pair_path, index=False)
        pd.DataFrame(matrix_cv, index=features, columns=features).to_csv(matrix_cv_path)
        pd.DataFrame(matrix_noncv, index=features, columns=features).to_csv(matrix_noncv_path)

        fig_dir = os.path.join(model.figure_dir, args.model)
        Path(fig_dir).mkdir(exist_ok=True, parents=True)
        plot_path = os.path.join(fig_dir,
                                 f'{args.model}_correlation_noncv_vs_cv.png')
        plot_cv_vs_noncv_heatmaps(
            matrix_cv, matrix_noncv, features, plot_path,
            subtitle=(f'model={args.model}  '
                      f'chunklen={model.chunklen}  latent_dim={model.latent_dim}  '
                      f'reg={model.reg_param}  windowed={model.apply_windowing}'))
        print('Saved:')
        for p in (per_pair_path, matrix_cv_path, matrix_noncv_path, plot_path):
            print(' ', p)
        return

    # ---- all Fig 1C groups ----
    cv_dir = args.cv_analysis_dir or os.path.join(args.dir, 'analysis',
                                                  'FeatureSpaceCorrelation')
    per_pair_df, group_df, matrix_cv, matrix_noncv = \
        run_all_fig1c_groups(model, cv_dir)

    out_dir = model.out_dir
    per_pair_path = os.path.join(out_dir, 'per_layer_pair_cv_vs_noncv.csv')
    group_path = os.path.join(out_dir, 'group_pair_cv_vs_noncv.csv')
    matrix_noncv_path = os.path.join(out_dir, 'group_matrix_noncv.csv')
    matrix_cv_path = os.path.join(out_dir, 'group_matrix_cv.csv')

    per_pair_df.to_csv(per_pair_path, index=False)
    group_df.to_csv(group_path, index=False)
    pd.DataFrame(matrix_noncv,
                 index=FIG1C_GROUP_ORDER,
                 columns=FIG1C_GROUP_ORDER).to_csv(matrix_noncv_path)
    pd.DataFrame(matrix_cv,
                 index=FIG1C_GROUP_ORDER,
                 columns=FIG1C_GROUP_ORDER).to_csv(matrix_cv_path)

    plot_path = os.path.join(model.figure_dir,
                             'featurespace_correlation_noncv_vs_cv.png')
    plot_cv_vs_noncv_heatmaps(
        matrix_cv, matrix_noncv, FIG1C_GROUP_ORDER, plot_path,
        subtitle=(f'chunklen={model.chunklen}  latent_dim={model.latent_dim}  '
                  f'reg={model.reg_param}  '
                  f'windowed={model.apply_windowing}'))

    print('Saved:')
    print(' ', per_pair_path)
    print(' ', group_path)
    print(' ', matrix_noncv_path)
    print(' ', matrix_cv_path)
    print(' ', plot_path)


if __name__ == '__main__':
    main()
