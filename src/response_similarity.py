"""
Run the response similarity analysis on fitted encoding model weights (Main Figure 4).

Computes, within each participant, the Spearman correlation between the encoding model
weights of a feature space averaged within each pair of functionally localized regions.

Usage (the calls that produced Main Figure 4 are in scripts/submit_response_similarity.sh):
    python -m src.response_similarity         --model vislang_gpt2_chunk120_eps0.3 --chunklen 120         --axes alexnet:alexnet_eps0.3_layer5,alexnet_eps0.3_layer6,alexnet_eps0.3_layer7         --filepath_tag alexnet_eps0.3         --average_posterior_anterior --no_split_hemi

The results are saved to <out_dir>/SecondLevelIndividual/ and replotted in
SecondLevel Analyses.ipynb (with load=True).
"""
import os
import matplotlib
import sys
import argparse

from src import secondlevel_individual_encoding
from src import helpers


os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


matplotlib.use('Agg')


def parse_axes(axes_args):
    """Parse CLI axes format into dict.

    Input:  ['alexnet:alexnet_eps0.3_layer6', 'GPT2:GPT2_chunk120_layer11,GPT2_chunk120_layer12']
    Output: {'alexnet': ['alexnet_eps0.3_layer6'], 'GPT2': ['GPT2_chunk120_layer11', 'GPT2_chunk120_layer12']}
    """
    axes = {}
    for item in axes_args:
        key, values = item.split(':', 1)
        axes[key] = values.split(',')
    return axes


def main():
    parser = argparse.ArgumentParser(
        description='Run response similarity analysis on encoding model weights'
    )

    # Paths
    parser.add_argument('--dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
                        help='Project root directory')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: <dir>/analysis)')
    parser.add_argument('--figure_dir', default=None,
                        help='Figure directory (default: <dir>/figures/subfigures)')

    # Response similarity parameters
    parser.add_argument('--response', default='weights',
                        help='Response type: weights, ind_product_measure, or predicted_time_series')
    parser.add_argument('--axes', nargs='+', required=True,
                        help='Feature axes in format key:val1,val2 (e.g., alexnet:alexnet_eps0.3_layer6 GPT2:GPT2_chunk120_layer12)')
    parser.add_argument('--filepath_tag', default='',
                        help='Tag appended to output filenames')
    parser.add_argument('--average_posterior_anterior', action='store_true',
                        help='Collapse posterior/anterior regions')
    parser.add_argument('--no_split_hemi', action='store_true',
                        help='Average the hemispheres instead of splitting left/right')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting')

    # Encoding model parameters
    parser.add_argument('--model', default='vislang_gpt2_chunk120_eps0.3',
                        help='Encoding model name')
    parser.add_argument('--mask', default='ISC',
                        help='Brain mask name')
    parser.add_argument('--space', default='MNI152NLin2009cAsym',
                        help='Brain space')
    parser.add_argument('--smoothing_fwhm', type=float, default=3.0,
                        help='Smoothing FWHM')
    parser.add_argument('--chunklen', type=int, default=120,
                        help='Chunk length')
    parser.add_argument('--perc_top_voxels', type=int, default=5,
                        help='Percentile of top voxels for localizer masks')
    parser.add_argument('--population', default='NT',
                        help='Subject population')

    args = parser.parse_args()

    # Resolve default paths
    if args.out_dir is None:
        args.out_dir = os.path.join(args.dir, 'analysis')
    if args.figure_dir is None:
        args.figure_dir = os.path.join(args.dir, 'figures', 'subfigures')

    split_hemi = not args.no_split_hemi
    axes = parse_axes(args.axes)
    plot = not args.no_plot

    print(f"\n{'='*70}")
    print(f"Response Similarity Analysis")
    print(f"{'='*70}")
    print(f"  Model: {args.model}")
    print(f"  Response: {args.response}")
    print(f"  Population: {args.population}")
    print(f"  Axes: {axes}")
    print(f"  Filepath tag: '{args.filepath_tag}'")
    print(f"  Split hemi: {split_hemi}")
    print(f"  Average posterior/anterior: {args.average_posterior_anterior}")
    print(f"  Plot: {plot}")
    print(f"  Perc top voxels: {args.perc_top_voxels}")
    print(f"{'='*70}\n")

    secondlevel_ind = secondlevel_individual_encoding.SecondLevelIndividual(args)

    print(f"  Subjects loaded: {len(secondlevel_ind.subjects.get('sherlock', []))}")

    secondlevel_ind.compute_response_similarity(
        load=False,
        plot=plot,
        response=args.response,
        split_hemi=split_hemi,
        axes=axes,
        filepath_tag=args.filepath_tag,
        average_posterior_anterior=args.average_posterior_anterior
    )

    print(f"\n{'='*70}")
    print(f"Response similarity analysis complete")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
