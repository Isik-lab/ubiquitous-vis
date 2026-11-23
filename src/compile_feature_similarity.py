import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from src import helpers
from src import plotting_helpers


def get_feature_names(model_name, features_override=None, include_scene_description=False):
    """Get feature names from model dict or override string."""
    if features_override is not None:
        if isinstance(features_override, str):
            features = [f.strip() for f in features_override.split(',')]
        else:
            features = list(features_override)
    else:
        models_dict = helpers.get_models_dict()
        features = models_dict[model_name]
    if include_scene_description:
        features = features + [f'scene_description_layer{layer}' for layer in range(1, 13)]
    return features


def compile_featurespace_correlation_results(features, analysis_dir, latent_dim=1, speech_masked=False):
    """Load pairwise CCA results from CSV files into an NxN matrix."""
    n = len(features)
    results_matrix = np.full((n, n), np.nan)

    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features):
            if feature1 == feature2:
                continue

            # Build filename
            base = f"{feature1}-{feature2}"
            if latent_dim is not None:
                base += f"_latent_dim-{latent_dim}"
            if speech_masked:
                base += "_speech-masked"
            filepath = os.path.join(analysis_dir, base + ".csv")

            try:
                result = pd.read_csv(filepath, header=None).iloc[0, 0]
            except Exception:
                result = np.nan

            if np.isnan(results_matrix[i][j]):
                results_matrix[i][j] = result
            if np.isnan(results_matrix[j][i]):
                results_matrix[j][i] = result

    return results_matrix


def plot_featurespace_correlation(features, results_matrix, output_filename, average_over_layers=True):
    """Plot feature similarity matrix as a lower-triangular heatmap."""
    cmap = plotting_helpers.get_cmaps()['matrix_green']
    colors_dict = plotting_helpers.get_colors_dict()

    feature_labels_dict = {
        'alexnet': 'image', 'alexnet_eps0.3': 'image',
        'alexnet_eps0.1': 'image', 'alexnet_eps0.2': 'image',
        'alexnet_eps0.4': 'image', 'alexnet_eps0.6': 'image',
        'motion': 'motion',
        'hubert': 'speech', 'hubert_chunk120': 'speech',
        'hubert_chunk60': 'speech', 'hubert_chunk80': 'speech',
        'sbert': 'sentence',
        'word2vec': 'word', 'word2vec_chunk120': 'word',
        'word2vec_chunk60': 'word', 'word2vec_chunk80': 'word',
        'GPT2_chunk120': 'GPT2', 'gpt2_chunk120': 'GPT2',
        'GPT2_chunk60': 'GPT2', 'GPT2_chunk80': 'GPT2',
        'social': 'social\ninteraction',
        'scene_description': 'scene\ndescription',
    }

    if average_over_layers:
        # Build DataFrame and group by model
        results_df = pd.DataFrame(results_matrix, columns=features)
        results_df['feature'] = [
            feature + '_layer' if 'layer' not in feature else feature
            for feature in features
        ]
        results_df['model'] = [label.split('_layer')[0] for label in results_df['feature']]
        results_df['layer'] = [label.split('layer')[1] for label in results_df['feature']]
        results_df['output'] = [label.split('_layer')[1] for label in results_df['feature']]
        results_df['model_output'] = [
            label.split('_layer')[0] if 'layer' in label.split('_')[1]
            else label.split('_')[0] + '_' + label.split('_')[1]
            for label in results_df['feature']
        ]
        results_df.columns = (
            results_df['model_output'].tolist()
            + ['feature', 'model', 'layer', 'output', 'model_output']
        )

        # Average over columns and rows of same model
        results_df = results_df.groupby(by='model_output', axis=0).mean()
        results_df = results_df.groupby(by=results_df.columns, axis=1).mean()

        # Build reduced feature list (preserving order, no duplicates)
        features_no_layer = [
            feature if 'layer' not in feature else feature.split('_layer')[0]
            for feature in features
        ]
        features = list(dict.fromkeys(features_no_layer))

        # Populate reduced matrix
        n = len(features)
        reduced_matrix = np.ones((n, n)) * 100
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if not np.isnan(results_df[f1][f2]):
                    reduced_matrix[i][j] = results_df[f1][f2]
                    reduced_matrix[j][i] = results_df[f1][f2]
        results_matrix = reduced_matrix

        # Mask upper triangle
        results_matrix[np.triu_indices(n, 0)] = np.nan

        font_size = 1.2 * n
        anchor = (-1.6, 0.7)
        plt.rcParams.update({'font.family': 'Arial', 'font.size': font_size})

        fig = plt.figure(figsize=(n * 1.5, n * 1.5))
        ax = fig.add_subplot(111)
        cax = ax.imshow(results_matrix, cmap=cmap, vmin=0, vmax=1)
        sns.despine(left=True, bottom=True)
        plt.title('')

        # Tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        feature_labels = [f.replace('_', ' ').replace(' layer', '').replace('1sent', '') for f in features]
        feature_labels_final = [feature_labels_dict.get(f, f.replace('_', ' ')) for f in features]
        ax.set_xticklabels(feature_labels_final[:-1] + [''], minor=False, rotation=90)
        ax.set_yticklabels([''] + feature_labels_final[1:], minor=False)

        # Color tick labels
        for label, xticklabel, yticklabel in zip(feature_labels, ax.get_xticklabels(), ax.get_yticklabels()):
            color = colors_dict.get(label, 'black')
            xticklabel.set_color(color)
            yticklabel.set_color(color)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                c = results_matrix[j, i]
                if not np.isnan(c):
                    ax.text(i, j, str(np.round(c, 2)), va='center', ha='center', fontweight='light')

    else:
        n = len(features)
        results_matrix[np.triu_indices(n, 0)] = np.nan

        font_size = 2.5 * n
        anchor = (-0.75, 0.8)
        plt.rcParams.update({'font.family': 'Arial', 'font.size': font_size})

        fig = plt.figure(figsize=(n * 1.5, n * 1.5))
        ax = fig.add_subplot(111)
        cax = ax.imshow(results_matrix, cmap=cmap, vmin=0, vmax=1)
        sns.despine(left=True, bottom=True)
        plt.title('')

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        # Model grouping labels and rectangles
        group_labels = [
            name.split('_')[0]
            if name.split('_')[0] in ['alexnet', 'hubert', 'sbert', 'SimCLR', 'GPT2']
            else ''
            for name in features
        ]

        current_label = group_labels[0]
        start_idx = 0
        for i, label in enumerate(group_labels):
            if label != current_label or i == len(group_labels) - 1:
                current_label = current_label.replace('1sent', '')
                end_idx = i if i == len(group_labels) - 1 else i - 1
                mid_point = (start_idx + end_idx) / 2
                shift_off_axis = 0.75
                ax.text(mid_point, n + shift_off_axis + 1, current_label,
                        ha='center', va='center', fontsize=font_size)
                ax.text(-shift_off_axis - 3, mid_point, current_label,
                        ha='center', va='center', fontsize=font_size)
                if current_label in ['alexnet', 'hubert', 'sbert']:
                    rect_h = Rectangle(
                        (start_idx - 0.2, n + shift_off_axis),
                        end_idx - start_idx + 0.2, 0,
                        linewidth=2, edgecolor='black', facecolor='none', clip_on=False)
                    ax.add_patch(rect_h)
                    rect_v = Rectangle(
                        (-shift_off_axis - 1, start_idx - 0.2),
                        0, end_idx - start_idx + 0.2,
                        linewidth=2, edgecolor='black', facecolor='none', clip_on=False)
                    ax.add_patch(rect_v)
                current_label = label
                start_idx = i

        # Strip model prefix from tick labels
        feature_labels = [
            name.replace('_', ' ')
                .replace('GPT2', '').replace('1sent', '')
                .replace('alexnet', '').replace('sbert', '')
                .replace('hubert', '').replace('SimCLR ', '')
                .replace('SLIP ', '').replace('attention', '')
                .replace('embedding', '').replace('layer', '')
            for name in features
        ]
        ax.set_xticklabels(feature_labels, minor=False, rotation=90)
        ax.set_yticklabels(feature_labels, minor=False)

    # Common settings
    ax.tick_params(axis='both', which='both', length=0)
    cbar = fig.colorbar(cax, label='Canonical correlation coefficient',
                        shrink=0.35, location='right', anchor=anchor)
    cbar.ax.yaxis.set_label_position('left')
    plt.xticks(rotation=90)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved: {output_filename}')


def main():
    parser = argparse.ArgumentParser(
        description='Compile and plot feature space similarity results.')
    parser.add_argument('--model-name', required=True, type=str,
                        help='Key into helpers.get_models_dict(), e.g. joint_chunk120')
    parser.add_argument('--features', type=str, default=None,
                        help='Comma-separated feature list override')
    parser.add_argument('--base-dir', type=str, default='../',
                        help='Project root directory')
    parser.add_argument('--latent-dim', type=int, default=1,
                        help='CCA latent dimension for filename lookup')
    parser.add_argument('--average-over-layers', action='store_true',
                        help='Average layers within each model group')
    parser.add_argument('--speech-masked', action='store_true',
                        help='Load speech-masked CSV variants')
    parser.add_argument('--include-scene-description', action='store_true',
                        help='Append scene_description_layer1..12 to feature list')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG filepath (auto-generated if omitted)')
    args = parser.parse_args()

    features = get_feature_names(args.model_name, args.features, args.include_scene_description)
    analysis_dir = os.path.join(args.base_dir, 'analysis', 'FeatureSpaceCorrelation')

    results_matrix = compile_featurespace_correlation_results(
        features, analysis_dir, args.latent_dim, args.speech_masked)

    if args.output is None:
        suffix = '_layers_averaged' if args.average_over_layers else ''
        masked_suffix = '_speech_masked' if args.speech_masked else ''
        scene_suffix = '_with_scene_description' if args.include_scene_description else ''
        args.output = os.path.join(
            args.base_dir, 'figures', 'main',
            f'featurespace_correlation{suffix}{masked_suffix}{scene_suffix}.png')

    plot_featurespace_correlation(features, results_matrix, args.output, args.average_over_layers)


if __name__ == '__main__':
    main()
