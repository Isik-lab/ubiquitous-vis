"""
Pooled PCA of encoding model weights across subjects (Main Figure 5).

Voxels from all subjects are pooled per ROI x feature space and a single PCA is fit, giving
population-level weight dimensions that are then interpreted with src.pc_interpretation.

Usage:
    # write analysis/PooledPCA/pooled_pca_combinations.tsv (one row per ROI x feature space;
    # '+' joins layers that are concatenated into one PCA)
    python -m src.pooled_pca combinations --model vislang_gpt2_chunk120_eps0.3 --feature_groups \
        motion "alexnet_eps0.3_layer5+alexnet_eps0.3_layer6+alexnet_eps0.3_layer7"

    # run one ROI x feature space (on a cluster, one per line of the combinations file, e.g. as an array job)
    python -m src.pooled_pca run --feature_spaces motion --roi MT --roi_type func --population NT
"""
import os
import sys
import csv
import argparse
import numpy as np

from src import weight_pca_analysis
from src import helpers


def combinations_main():
    parser = argparse.ArgumentParser(
        description='Generate pooled PCA combinations from saved features'
    )
    parser.add_argument('--dir', default=helpers.PROJECT_DIR,
                       help='Project root directory')
    parser.add_argument('--model', default='',
                       help='Encoding model type')
    parser.add_argument('--smoothing_fwhm', type=int, default=0,
                       help='Smoothing FWHM')
    parser.add_argument('--chunklen', type=int, default=20,
                       help='Chunk length')
    parser.add_argument('--mask', default='cortex',
                       help='Brain mask name')
    parser.add_argument('--feature_groups', nargs='+', default=None,
                       help="Explicit feature-space entries to write to the combinations file. "
                            "Use '+' to concatenate multiple features in one entry "
                            "(e.g., 'alexnet_eps0.3_layer5+alexnet_eps0.3_layer6'). "
                            "If omitted, one entry per saved feature is generated.")
    parser.add_argument('--func_rois', nargs='+', default=['MT', 'STS', 'temporal_language', 'frontal'],
                       help='Functional ROIs to include')
    parser.add_argument('--anat_rois', nargs='+', default=['MT', 'STS', 'temporal_language', 'frontal'],
                       help='Anatomical ROIs to include')

    args = parser.parse_args()

    # Path to weight features file
    enc_file_label = (f'_encoding_model-{args.model}'
                     f'_smoothingfwhm-{args.smoothing_fwhm}'
                     f'_chunklen-{args.chunklen}'
                     f'_mask-{args.mask}')

    features_csv = os.path.join(
        args.dir, 'analysis', 'EncodingModel', 'features',
        f'encoding_model-{args.model}_weight_features.csv'
    )

    if not os.path.exists(features_csv):
        print(f"ERROR: Weight features file not found: {features_csv}")
        print(f"Expected location: analysis/EncodingModel/features/")
        sys.exit(1)

    # Load saved features
    with open(features_csv, 'r') as f:
        reader = csv.reader(f)
        saved_features = next(reader)  # First row contains feature names

    print(f"Found {len(saved_features)} features with saved weights:")
    for i, feat in enumerate(saved_features, 1):
        print(f"  {i:2d}. {feat}")

    # Determine the list of feature-space entries to write.
    # Each entry may be a single feature or several joined by '+'.
    if args.feature_groups is not None:
        feature_entries = args.feature_groups
        # Validate every component has saved weights
        for entry in feature_entries:
            for component in entry.split('+'):
                if component not in saved_features:
                    print(f"ERROR: '{component}' (in entry '{entry}') is not in saved features")
                    sys.exit(1)
        print(f"\nUsing {len(feature_entries)} feature-space entries (from --feature_groups):")
        for e in feature_entries:
            print(f"  {e}")
    else:
        feature_entries = saved_features

    func_rois = args.func_rois
    anat_rois = args.anat_rois

    # Generate all combinations
    combinations = []
    for roi in func_rois:
        for feature in feature_entries:
            combinations.append({
                'roi': roi,
                'roi_type': 'func',
                'feature_space': feature
            })
    for roi in anat_rois:
        for feature in feature_entries:
            combinations.append({
                'roi': roi,
                'roi_type': 'anat',
                'feature_space': feature
            })

    total_rois = len(func_rois) + len(anat_rois)
    print(f"\nGenerated {len(combinations)} ROI × feature combinations:")
    print(f"  Functional ROIs: {len(func_rois)} ({', '.join(func_rois)})")
    print(f"  Anatomical ROIs: {len(anat_rois)} ({', '.join(anat_rois)})")
    print(f"  Feature entries: {len(feature_entries)}")
    print(f"  Total: {total_rois} × {len(feature_entries)} = {len(combinations)}")

    # Save to TSV
    output_dir = os.path.join(args.dir, 'analysis', 'PooledPCA')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'pooled_pca_combinations.tsv')

    with open(output_file, 'w') as f:
        for combo in combinations:
            f.write(f"{combo['roi']}\t{combo['roi_type']}\t{combo['feature_space']}\n")

    print(f"\nSaved combinations to: {output_file}")
    print(f"\nRun python -m src.pooled_pca run once per line ({len(combinations)} in total), e.g. as an array job")

    # Print summary by ROI
    print("\nCombinations per ROI:")
    for roi_type_label, roi_list in [('func', func_rois), ('anat', anat_rois)]:
        for roi in roi_list:
            count = sum(1 for c in combinations if c['roi'] == roi and c['roi_type'] == roi_type_label)
            print(f"  {roi:20s} ({roi_type_label:4s}): {count:3d} features")

    # Print a few example combinations
    print("\nFirst 5 combinations:")
    for i, combo in enumerate(combinations[:5], 1):
        print(f"  {i}. {combo['roi']:20s} {combo['roi_type']:5s} {combo['feature_space']}")

    print("\nLast 5 combinations:")
    for i, combo in enumerate(combinations[-5:], len(combinations)-4):
        print(f"  {i}. {combo['roi']:20s} {combo['roi_type']:5s} {combo['feature_space']}")


def run_pooled_pca(feature_space, roi, roi_type, subjects, args):
    """
    Run pooled PCA for one ROI × feature combination across subjects.

    Args:
        feature_space: Feature space name (e.g., 'sbert_layer10') OR list of
            feature space names to concatenate into a single PCA.
        roi: ROI name (e.g., 'STS', 'MT')
        roi_type: 'func' or 'anat'
        subjects: List of subject IDs
        args: Namespace with configuration

    Returns:
        dict with status and results
    """
    fs_label = '+'.join(feature_space) if isinstance(feature_space, (list, tuple)) else feature_space

    print(f"\n{'='*70}")
    print(f"Pooled PCA Analysis")
    print(f"{'='*70}")
    print(f"  Feature space: {fs_label}")
    print(f"  ROI: {roi} ({roi_type})")
    print(f"  Subjects: {len(subjects)}")
    print(f"  n_components: {args.n_components}")
    print(f"  min_performance: {args.min_performance}")

    try:
        # Initialize WeightPCAAnalysis
        weight_pca = weight_pca_analysis.WeightPCAAnalysis(args)

        # Compute pooled PCA across all subjects
        pooled_results = weight_pca.compute_pooled_pca_across_subjects(
            subjects=subjects,
            roi=roi,
            feature_space=feature_space,
            n_components=args.n_components,
            roi_type=roi_type,
            min_performance=args.min_performance,
            whiten=False
        )

        # Save results
        weight_pca.save_pooled_pca_results(pooled_results)

        return {
            'status': 'success',
            'feature_space': fs_label,
            'roi': roi,
            'roi_type': roi_type,
            'n_subjects': len(pooled_results['subjects']),
            'total_voxels': pooled_results['total_voxels'],
            'n_components': pooled_results['n_components'],
            'cumulative_variance': pooled_results['cumulative_variance'][-1]
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        return {
            'status': 'failed',
            'feature_space': fs_label,
            'roi': roi,
            'roi_type': roi_type,
            'error': str(e)
        }


def run_main():
    parser = argparse.ArgumentParser(
        description='Run pooled PCA analysis across subjects'
    )

    # Paths
    parser.add_argument('--dir', default=helpers.PROJECT_DIR,
                       help='Project root directory')
    parser.add_argument('--out_dir', default='analysis',
                       help='Output directory (relative to --dir)')

    # Analysis specification
    parser.add_argument('--feature_space', type=str,
                       help='Single feature space (e.g., sbert_layer10). If not specified and '
                            '--feature_spaces not given, runs all saved features one at a time.')
    parser.add_argument('--feature_spaces', nargs='+', default=None,
                       help='Multiple feature spaces to concatenate into one PCA '
                            '(e.g., --feature_spaces alexnet_eps0.3_layer5 alexnet_eps0.3_layer6). '
                            'Overrides --feature_space.')
    parser.add_argument('--roi', type=str,
                       help='ROI name (e.g., STS, MT). If not specified, runs all ROIs.')
    parser.add_argument('--roi_type', default='func', choices=['func', 'anat'],
                       help='ROI type: func or anat')

    # Subject selection
    parser.add_argument('--population', default='NT',
                       choices=['NT'],
                       help='Subject population to include')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Specific subjects to include (overrides --population)')

    # PCA parameters
    parser.add_argument('--n_components', type=float, default=0.95,
                       help='Number of PCs or variance to retain (default: 0.95)')
    parser.add_argument('--min_performance', type=float, default=0.0,
                       help='Minimum encoding performance threshold for voxel inclusion')

    # Encoding model parameters (for loading weights)
    parser.add_argument('--model', default='vislang_gpt2_chunk60',
                       help='Encoding model type')
    parser.add_argument('--smoothing_fwhm', type=float, default=3.0,
                       help='Smoothing FWHM')
    parser.add_argument('--chunklen', type=int, default=60,
                       help='Chunk length')
    parser.add_argument('--mask', default='ISC',
                       help='Brain mask name')
    parser.add_argument('--space', default='MNI152NLin2009cAsym_res-2',
                       help='Brain space')

    args = parser.parse_args()

    # Resolve paths
    args.out_dir = os.path.join(args.dir, args.out_dir)

    # Get subject list
    if args.subjects is not None:
        subjects = args.subjects
        print(f"Using {len(subjects)} specified subjects")
    else:
        subjects_dict = helpers.get_subjects(args.population)
        subjects = subjects_dict['sherlock']
        print(f"Using {len(subjects)} subjects from population: {args.population}")

    # Determine feature-space iteration:
    # - If --feature_spaces (plural) is given, treat as a SINGLE concatenated PCA run
    # - Else if --feature_space (singular) is given, run just that one
    # - Else run one PCA per saved feature space
    if args.feature_spaces is not None:
        feature_space_items = [args.feature_spaces]  # one item: the list to concatenate
        print(f"Concatenating {len(args.feature_spaces)} feature spaces into one PCA: {args.feature_spaces}")
    elif args.feature_space is not None:
        feature_space_items = [args.feature_space]
    else:
        weight_pca = weight_pca_analysis.WeightPCAAnalysis(args)
        feature_space_items = weight_pca.saved_features
        print(f"Processing all {len(feature_space_items)} feature spaces")

    # Get ROIs to process
    if args.roi is not None:
        rois = [args.roi]
    else:
        rois = ['MT', 'STS', 'temporal_language', 'frontal']
        print(f"Processing all {len(rois)} ROIs")

    # Run pooled PCA for each combination
    results = []

    for feature_space in feature_space_items:
        for roi in rois:
            result = run_pooled_pca(
                feature_space=feature_space,
                roi=roi,
                roi_type=args.roi_type,
                subjects=subjects,
                args=args
            )
            results.append(result)

            print(f"\nStatus: {result['status']}")
            if result['status'] == 'success':
                print(f"  Subjects: {result['n_subjects']}")
                print(f"  Total voxels: {result['total_voxels']}")
                print(f"  Components: {result['n_components']}")
                print(f"  Variance explained: {result['cumulative_variance']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Pooled PCA Analysis Complete")
    print(f"{'='*70}")
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(results)}")


if __name__ == '__main__':
    commands = {'combinations': combinations_main, 'run': run_main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit('usage: python -m ' + __spec__.name + ' {combinations|run} [options]')
    commands[sys.argv.pop(1)]()
