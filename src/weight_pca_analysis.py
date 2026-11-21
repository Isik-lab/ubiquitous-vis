"""
Huth 2016 Style PCA Analysis of Encoding Model Weights

This module performs PCA on encoding model weights following Huth et al. (2016) methodology:
analyzing voxels as observations and features as variables to discover principal semantic
dimensions in brain representations.

Reference:
Huth, A. G. et al., "Natural speech reveals the semantic maps that tile human cerebral cortex"
Nature 532, 453-458 (2016).
"""

import os
import csv
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py

import nibabel
import nilearn
from nilearn import plotting

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from src import helpers
from src import plotting_helpers
from src import encoding
from src.group_pca_analysis import pooled_label

plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})


# Available vision feature spaces for the "Image Model" column of the pooled PCA summaries.
IMAGE_MODEL_SPECS = {
    'alexnet': 'alexnet_eps0.3_layer5+alexnet_eps0.3_layer6+alexnet_eps0.3_layer7',
    'simclr_embedding': ('SimCLR_embedding_eps0.3_layer10+'
                        'SimCLR_embedding_eps0.3_layer11+'
                        'SimCLR_embedding_eps0.3_layer12'),
    'simclr_attention': ('SimCLR_attention_eps0.3_layer10+'
                        'SimCLR_attention_eps0.3_layer11+'
                        'SimCLR_attention_eps0.3_layer12'),
}


class WeightPCAAnalysis:
    """
    Perform Huth 2016 style PCA on encoding model weights.

    PCA is performed on [voxels × features] matrices where:
    - Rows (observations): voxels from a specific ROI
    - Columns (variables): feature dimensions from a feature space
    - Analysis goal: Discover principal semantic dimensions encoded by voxels
    """

    def __init__(self, args):
        """Initialize WeightPCAAnalysis with paths and parameters."""
        self.dir = args.dir
        self.data_dir = os.path.join(args.dir, 'data')
        self.enc_dir = os.path.join(args.out_dir, 'EncodingModel')
        self.out_dir = os.path.join(args.out_dir, 'WeightPCA')

        self.model = args.model
        self.smoothing_fwhm = args.smoothing_fwhm
        self.chunklen = args.chunklen
        self.mask_name = args.mask
        self.space = args.space
        self.population = args.population

        # File label for encoding files
        self.enc_file_label = (f'_encoding_model-{self.model}'
                              f'_smoothingfwhm-{self.smoothing_fwhm}'
                              f'_chunklen-{self.chunklen}'
                              f'_mask-{self.mask_name}')

        # Brain parameters
        self.brain_shape = (97, 115, 97)
        self.affine = None

        # ISC mask (voxels where weights are saved)
        self.isc_mask = None
        self.isc_voxel_indices = None

        # ROI to GLM contrast mapping (functional ROIs)
        self.roi_glm_mapping = {
            'MT': ('SIpointlights', 'interact&no_interact', ['MT']),
            'STS': ('SIpointlights', 'interact-no_interact', ['pSTS', 'aSTS']),
            'temporal_language': ('language', 'intact-degraded', ['pTemp', 'aTemp']),
            'frontal': ('language', 'intact-degraded', ['frontal'])
        }

        # Anatomical parcel mapping
        self.roi_anatomical_mapping = {
            'MT': 'MT',                           # MT.nii.gz
            'STS': 'STS',                         # lSTS + rSTS combined
            'lSTS': 'left-STS',                   # Left STS only
            'rSTS': 'right-STS',                  # Right STS only
            'temporal_language': 'temporal_language',  # SN220 masks 4,5,6,10,11,12
            'frontal': 'frontal',                 # SN220 masks 1,2,3,7,8,9
            'pTemp': 'pTemp',                     # SN220 masks 5,6,11,12
            'aTemp': 'aTemp',                     # SN220 masks 4,10
        }

        # Load saved feature list
        self.saved_features = self.load_saved_feature_list()
        print(f"Found {len(self.saved_features)} saved features: {self.saved_features}")

        # Create output directories
        self._create_output_dirs()

        # Caches
        self.cached_masks = {}
        self.cached_performance = {}

    def _create_output_dirs(self):
        """Create output directory structure."""
        subdirs = ['models', 'scores', 'loadings', 'variance',
                  'figures/variance_explained', 'figures/feature_loadings',
                  'figures/brain_maps', 'summary']
        for subdir in subdirs:
            Path(os.path.join(self.out_dir, subdir)).mkdir(exist_ok=True, parents=True)

    # ==================== Feature Availability ====================

    def load_saved_feature_list(self):
        """
        Load list of features that have saved weights.

        Returns:
            list: Feature space names that have weights saved
        """
        features_csv = os.path.join(
            self.enc_dir, 'features',
            f'encoding_model-{self.model}_weight_features.csv'
        )

        if not os.path.exists(features_csv):
            print(f"WARNING: Weight features file not found: {features_csv}")
            return []

        with open(features_csv, 'r') as f:
            reader = csv.reader(f)
            features = next(reader)  # Read first row

        return features

    def check_feature_has_weights(self, feature_space):
        """Check if a feature space has saved weights."""
        return feature_space in self.saved_features

    # ==================== Data Loading ====================

    def load_weight_matrix(self, subject):
        """
        Load complete weight matrix from HDF5.

        Args:
            subject: Subject ID (e.g., 'sub-05')

        Returns:
            weights: np.ndarray, shape (n_features_total, n_voxels)
        """
        weights_path = os.path.join(
            self.enc_dir, 'weights',
            f"{subject}{self.enc_file_label}_measure-weights_raw.h5"
        )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weight file not found: {weights_path}")

        print(f"  Loading weights from {os.path.basename(weights_path)}")

        with h5py.File(weights_path, 'r') as f:
            weights = f['weights'][()]  # Shape: (n_features, n_voxels)

            # Store affine if available
            if self.affine is None and 'affine' in f.attrs:
                self.affine = f.attrs['affine']

        return weights

    def extract_feature_block(self, weights, feature_space):
        """
        Extract weights for a specific feature space from concatenated matrix.

        Args:
            weights: Full weight matrix (n_features_total, n_voxels)
            feature_space: Target feature space (e.g., 'sbert_layer10')

        Returns:
            feature_weights: np.ndarray, shape (n_dims_feature, n_voxels)
        """
        # Feature dimensions from save_weight_subset.py
        N_FEATURES_DICT = {
            'motion': 2530,
            'alexnet_layer1': 6480,
            'alexnet_layer2': 6480,
            'alexnet_layer3': 6480,
            'alexnet_layer4': 6480,
            'alexnet_layer5': 6480,
            'alexnet_layer6': 4096,
            'alexnet_layer7': 4096,
            'word2vec': 300,
            'sbert_layer10': 768,
            'sbert_layer11': 768,
            'sbert_layer12': 768,
        }

        if feature_space not in self.saved_features:
            raise ValueError(f"Feature {feature_space} not in saved features list")

        # Compute feature spans based on saved feature order
        spans = {}
        cur = 0
        for fname in self.saved_features:
            if fname not in N_FEATURES_DICT:
                raise KeyError(f"Feature '{fname}' not in N_FEATURES_DICT.")
            n = N_FEATURES_DICT[fname]
            spans[fname] = (cur, cur + n)
            cur += n

        # Extract the specific feature block
        start_idx, end_idx = spans[feature_space]
        feature_weights = weights[start_idx:end_idx, :]

        print(f"  Extracted {feature_space}: shape {feature_weights.shape} (indices {start_idx}:{end_idx})")

        return feature_weights

    # ==================== Voxel Selection ====================

    def load_isc_mask(self):
        """
        Load ISC mask (defines which voxels have weights saved).

        Uses helpers.load_mask() method to get the proper ISC mask.

        Returns:
            isc_mask: Binary 3D array
            isc_voxel_indices: Flat indices in brain volume
        """
        if self.isc_mask is not None:
            return self.isc_mask, self.isc_voxel_indices

        # Load ISC mask using helpers method
        # This loads: sub-NT_brain2brain_correlation_smoothingfwhm-6.0_chunklen-20_measure-perf_p_fdr.nii.gz
        isc_mask_path = os.path.join(
            self.dir, 'analysis', 'SecondLevelGroup', 'performance',
            'sub-NT_brain2brain_correlation_smoothingfwhm-6.0_chunklen-20_measure-perf_p_fdr.nii.gz'
        )

        if not os.path.exists(isc_mask_path):
            raise FileNotFoundError(f"ISC mask not found at {isc_mask_path}")

        print(f"Loading ISC mask from {isc_mask_path}")
        isc_img = nibabel.load(isc_mask_path)
        self.isc_mask = isc_img.get_fdata() > 0
        self.isc_voxel_indices = np.where(self.isc_mask.ravel())[0]

        if self.affine is None:
            self.affine = isc_img.affine

        print(f"ISC mask has {len(self.isc_voxel_indices)} voxels")
        return self.isc_mask, self.isc_voxel_indices

    def load_glm_localizer_mask(self, subject, roi):
        """
        Load GLM-based localizer mask for ROI.

        Args:
            subject: Subject ID
            roi: ROI name ('MT', 'STS', 'temporal_language', 'frontal')

        Returns:
            voxel_mask: Binary 3D array (brain volume)
        """
        cache_key = (subject, roi)
        if cache_key in self.cached_masks:
            return self.cached_masks[cache_key]

        if roi not in self.roi_glm_mapping:
            raise ValueError(f"Unknown ROI: {roi}")

        glm_task, contrast, mask_names = self.roi_glm_mapping[roi]

        # Combine masks if multiple (e.g., pSTS + aSTS)
        combined_mask = np.zeros(self.brain_shape, dtype=bool)

        for mask_name in mask_names:
            mask_path = os.path.join(
                self.out_dir.replace('WeightPCA', 'SecondLevelIndividual'),
                'localizer_masks',
                f"{subject}_smoothingfwhm-{self.smoothing_fwhm}_mask-{mask_name}"
                f"_glm_loc-{contrast}_run-all_binary.nii.gz"
            )

            if not os.path.exists(mask_path):
                print(f"  WARNING: Localizer mask not found: {mask_path}")
                continue

            mask_img = nibabel.load(mask_path)
            mask_data = mask_img.get_fdata()
            combined_mask = combined_mask | (mask_data > 0)

            # Store affine
            if self.affine is None:
                self.affine = mask_img.affine

        self.cached_masks[cache_key] = combined_mask
        return combined_mask

    def load_anatomical_mask(self, subject, roi):
        """
        Load anatomical parcel mask for ROI.

        Args:
            subject: Subject ID (not used - anatomical masks same across subjects)
            roi: ROI name from roi_anatomical_mapping

        Returns:
            voxel_mask: Binary 3D array (brain volume)
        """
        cache_key = ('anat', roi)
        if cache_key in self.cached_masks:
            return self.cached_masks[cache_key]

        if roi not in self.roi_anatomical_mapping:
            raise ValueError(f"Unknown anatomical ROI: {roi}")

        mask_name = self.roi_anatomical_mapping[roi]

        # Create temporary helpers object to access load_mask
        class TempHelpers:
            def __init__(self, dir_path, brain_shape):
                self.dir = dir_path
                self.brain_shape = brain_shape

        temp_helpers = TempHelpers(self.dir, self.brain_shape)
        mask_img = helpers.load_mask(temp_helpers, mask_name)

        # Extract mask data
        mask_data = mask_img.get_fdata() > 0

        # Store affine if not set
        if self.affine is None:
            self.affine = mask_img.affine

        self.cached_masks[cache_key] = mask_data
        print(f"  Loaded anatomical mask '{roi}': {np.sum(mask_data)} voxels")
        return mask_data

    def load_roi_mask(self, subject, roi, roi_type='func'):
        """
        Unified interface to load ROI mask (functional or anatomical).

        Args:
            subject: Subject ID
            roi: ROI name
            roi_type: 'func' or 'anat'

        Returns:
            voxel_mask: Binary 3D array
        """
        if roi_type == 'func':
            return self.load_glm_localizer_mask(subject, roi)
        elif roi_type == 'anat':
            return self.load_anatomical_mask(subject, roi)
        else:
            raise ValueError(f"Unknown roi_type: {roi_type}. Must be 'func' or 'anat'")

    def load_encoding_performance(self, subject):
        """
        Load encoding model performance map.

        Args:
            subject: Subject ID

        Returns:
            perf_map: 3D array of performance values
        """
        if subject in self.cached_performance:
            return self.cached_performance[subject]

        perf_path = os.path.join(
            self.enc_dir, 'performance',
            f"{subject}{self.enc_file_label}_measure-perf_raw.nii.gz"
        )

        if not os.path.exists(perf_path):
            print(f"  WARNING: Performance file not found: {perf_path}")
            return None

        perf_img = nibabel.load(perf_path)
        perf_map = perf_img.get_fdata()

        self.cached_performance[subject] = perf_map
        return perf_map

    def select_voxels_roi_filtered(self, subject, roi, roi_type='func', min_performance=0.0):
        """
        Select voxels using ROI mask (functional or anatomical) + performance filter.

        Args:
            subject: Subject ID
            roi: ROI name
            roi_type: 'func' (GLM localizer) or 'anat' (anatomical parcel)
            min_performance: Minimum encoding performance threshold

        Returns:
            voxel_mask: Binary 3D array (full brain space)
            voxel_indices_isc: 1D array of indices in ISC mask space (for indexing weights)
        """
        # Load ISC mask
        isc_mask, isc_voxel_indices = self.load_isc_mask()

        # Load ROI mask (functional or anatomical)
        roi_mask = self.load_roi_mask(subject, roi, roi_type)

        # Load encoding performance
        perf_map = self.load_encoding_performance(subject)

        if perf_map is None:
            # Use only ROI mask if performance not available
            voxel_mask = roi_mask & isc_mask
        else:
            # Combine: ROI mask AND performance > threshold AND ISC mask
            voxel_mask = roi_mask & (perf_map > min_performance) & isc_mask

        # Get flat indices in full brain space
        voxel_indices_brain = np.where(voxel_mask.ravel())[0]

        # Map to ISC space (indices into weight matrix)
        # Create mapping from brain indices to ISC indices
        brain_to_isc = np.full(np.prod(self.brain_shape), -1, dtype=int)
        brain_to_isc[isc_voxel_indices] = np.arange(len(isc_voxel_indices))

        voxel_indices_isc = brain_to_isc[voxel_indices_brain]

        # Remove any invalid indices (shouldn't happen if masks are correct)
        valid = voxel_indices_isc >= 0
        voxel_indices_isc = voxel_indices_isc[valid]

        n_voxels = len(voxel_indices_isc)
        print(f"  Selected {n_voxels} voxels in {roi} ({roi_type} + perf > {min_performance})")

        return voxel_mask, voxel_indices_isc

    # Backward compatibility alias
    def select_voxels_glm_filtered(self, subject, roi, min_performance=0.0):
        """Deprecated: Use select_voxels_roi_filtered instead."""
        return self.select_voxels_roi_filtered(subject, roi, 'func', min_performance)

    # ==================== PCA Computation ====================

    def compute_pca(self, weight_matrix_roi, n_components=0.95, whiten=False):
        """
        Perform PCA on [voxels × features] weight matrix.

        Following Huth et al. 2016:
        - Voxels are observations (rows)
        - Features are variables (columns)
        - Normalize within each voxel (z-score across features)
        - Extract PC scores (voxel coordinates in semantic space)
        - Extract PC loadings (feature contributions)

        Args:
            weight_matrix_roi: np.ndarray, shape (n_voxels, n_features)
            n_components: int, float, or 'mle'
                - int: exact number of components
                - float (0-1): retain this much variance
                - 'mle': use Minka's MLE
            whiten: bool, whether to whiten PC scores

        Returns:
            pca_results: dict with PCA outputs
        """
        n_voxels, n_features = weight_matrix_roi.shape
        print(f"  Computing PCA on {n_voxels} voxels × {n_features} features")

        # Within-voxel normalization (z-score each voxel across features)
        # For each voxel (row): subtract mean and divide by std across features
        voxel_means = np.mean(weight_matrix_roi, axis=1, keepdims=True)
        voxel_stds = np.std(weight_matrix_roi, axis=1, keepdims=True)
        # Avoid division by zero for voxels with constant weights
        voxel_stds = np.where(voxel_stds == 0, 1.0, voxel_stds)
        X_normalized = (weight_matrix_roi - voxel_means) / voxel_stds

        print(f"  Applied within-voxel normalization (z-score across features)")
        print(f"  Each voxel now has mean=0, std=1 across its feature weights")

        # Fit PCA on within-voxel normalized data
        pca = PCA(n_components=n_components, whiten=whiten, svd_solver='auto')
        pc_scores = pca.fit_transform(X_normalized)

        # Package results
        pca_results = {
            'pca_model': pca,
            'voxel_means': voxel_means,
            'voxel_stds': voxel_stds,
            'pc_scores': pc_scores,  # (n_voxels, n_components)
            'pc_loadings': pca.components_,  # (n_components, n_features)
            'variance_explained': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_components': pca.n_components_,
            'singular_values': pca.singular_values_,
            'mean': pca.mean_
        }

        print(f"  Retained {pca.n_components_} components "
              f"({pca_results['cumulative_variance'][-1]:.3f} cumulative variance)")

        return pca_results

    # ==================== Analysis Workflow ====================

    def analyze_feature_space(self, subject, feature_space, roi, roi_type='func',
                             n_components=0.95, save_outputs=True):
        """
        Complete PCA analysis pipeline for one feature space × ROI combination.

        Args:
            subject: Subject ID
            feature_space: Feature space name
            roi: ROI name
            roi_type: 'func' (functional GLM) or 'anat' (anatomical parcel)
            n_components: PCA component specification
            save_outputs: Whether to save results

        Returns:
            pca_results: dict or None if analysis failed
        """
        print(f"\nAnalyzing: {subject} | {feature_space} | {roi_type} | {roi}")

        # Check if feature has weights
        if not self.check_feature_has_weights(feature_space):
            print(f"  SKIP: {feature_space} does not have saved weights")
            return None

        try:
            # Load weight matrix
            weights = self.load_weight_matrix(subject)

            # Extract feature block
            feature_weights = self.extract_feature_block(weights, feature_space)

            # Select voxels in ROI (returns indices in ISC space)
            voxel_mask, voxel_indices_isc = self.select_voxels_roi_filtered(
                subject, roi, roi_type=roi_type, min_performance=0.0
            )

            if len(voxel_indices_isc) < 10:
                print(f"  SKIP: Too few voxels ({len(voxel_indices_isc)}) in {roi} ({roi_type})")
                return None

            # Subset to ROI voxels and transpose
            # feature_weights shape: (n_features, n_voxels_isc)
            # voxel_indices_isc: indices into the ISC voxel dimension
            weights_roi = feature_weights[:, voxel_indices_isc].T  # (n_voxels_roi, n_features)

            # Compute PCA
            pca_results = self.compute_pca(weights_roi, n_components=n_components)

            # Save outputs
            if save_outputs:
                self.save_pca_results(pca_results, subject, feature_space, roi, roi_type, voxel_mask)
                self._generate_visualizations(pca_results, subject, feature_space, roi, roi_type)

            return pca_results

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_all_feature_spaces(self, subjects=None, feature_spaces=None,
                                   rois=None, roi_types=None, n_components=0.95):
        """
        Batch process PCA for all feature spaces × ROIs × roi_types × subjects.

        Args:
            subjects: list of subject IDs (default: detect from data)
            feature_spaces: list of feature names (default: all saved)
            rois: list of ROI names (default: all 4)
            roi_types: list of ROI types ['func', 'anat'] (default: ['func'])
            n_components: variance threshold or number of components

        Returns:
            summary_df: DataFrame with analysis summary
        """
        # Set defaults
        if feature_spaces is None:
            feature_spaces = self.saved_features
        if rois is None:
            rois = ['MT', 'STS', 'temporal_language', 'frontal']
        if roi_types is None:
            roi_types = ['func']  # Default to functional only for backward compatibility
        if subjects is None:
            # Detect subjects from weight files
            subjects = self._get_available_subjects()

        print(f"\nStarting batch analysis:")
        print(f"  Subjects: {len(subjects)}")
        print(f"  Feature spaces: {len(feature_spaces)}")
        print(f"  ROI types: {roi_types}")
        print(f"  ROIs: {len(rois)}")
        print(f"  Total analyses: {len(subjects) * len(feature_spaces) * len(roi_types) * len(rois)}")

        # Run analyses
        log_entries = []

        for subject in subjects:
            for feature_space in feature_spaces:
                for roi_type in roi_types:
                    for roi in rois:
                        pca_results = self.analyze_feature_space(
                            subject, feature_space, roi, roi_type=roi_type,
                            n_components=n_components
                        )

                        if pca_results is not None:
                            log_entries.append({
                                'subject': subject,
                                'feature_space': feature_space,
                                'roi': roi,
                                'roi_type': roi_type,
                                'n_voxels': pca_results['pc_scores'].shape[0],
                                'n_features': pca_results['pc_loadings'].shape[1],
                                'n_components': pca_results['n_components'],
                                'variance_explained': pca_results['cumulative_variance'][-1],
                                'status': 'success',
                                'date': datetime.now().isoformat()
                            })
                        else:
                            log_entries.append({
                                'subject': subject,
                                'feature_space': feature_space,
                                'roi': roi,
                                'roi_type': roi_type,
                                'status': 'failed',
                                'date': datetime.now().isoformat()
                            })

        # Save log
        summary_df = pd.DataFrame(log_entries)
        summary_path = os.path.join(self.out_dir, 'summary', 'analysis_log.csv')
        summary_df.to_csv(summary_path, index=False)

        print(f"\nBatch analysis complete!")
        print(f"  Success: {(summary_df['status'] == 'success').sum()}")
        print(f"  Failed: {(summary_df['status'] == 'failed').sum()}")
        print(f"  Log saved: {summary_path}")

        return summary_df

    def compute_pooled_pca_across_subjects(self, subjects, roi, feature_space,
                                          n_components=0.95, roi_type='func',
                                          min_performance=0.0, whiten=False):
        """
        Compute PCA on pooled voxels from all subjects.

        This method identifies shared semantic dimensions across the population by:
        1. Loading encoding weights from all subjects for a specific ROI
        2. Stacking all voxels (across subjects) into one matrix
        3. Computing PCA on the pooled data
        4. Returning shared PC loadings that all subjects can use

        Args:
            subjects: list of subject IDs
            roi: ROI name (e.g., 'STS', 'MT', 'temporal_language')
            feature_space: Feature space name (e.g., 'sbert_layer10') OR list of
                feature space names to concatenate along the feature axis
                (e.g., ['alexnet_eps0.3_layer5','alexnet_eps0.3_layer6']).
            n_components: int, float, or 'mle' for PCA components
            roi_type: 'func' (functional GLM) or 'anat' (anatomical parcel)
            min_performance: Minimum encoding performance threshold for voxel selection
            whiten: bool, whether to whiten PC scores

        Returns:
            pooled_pca_results: dict with:
                - loadings: (K, D) shared PC loadings in feature space
                - explained_var: (K,) variance explained by each PC
                - cumulative_variance: (K,) cumulative variance explained
                - n_voxels_per_subject: dict {subject: n_voxels}
                - total_voxels: int, total number of voxels pooled
                - n_components: int, number of PCs retained
                - n_features: int, feature dimensionality
                - subjects: list of subjects included
                - roi: ROI name
                - roi_type: ROI type
                - feature_space: feature space label (joined by '+' if multiple)
                - feature_spaces: list of feature space names included
        """
        # Normalize feature_space to a list
        if isinstance(feature_space, str):
            feature_spaces = [feature_space]
        else:
            feature_spaces = list(feature_space)
        feature_space_label = '+'.join(feature_spaces)

        print(f"\n{'='*60}")
        print(f"Computing Pooled PCA Across Subjects")
        print(f"{'='*60}")
        print(f"  Feature space(s): {feature_spaces}")
        print(f"  ROI: {roi} ({roi_type})")
        print(f"  Subjects: {len(subjects)}")
        print(f"  Min performance: {min_performance}")

        # Check each feature has weights
        for fs in feature_spaces:
            if not self.check_feature_has_weights(fs):
                raise ValueError(f"{fs} does not have saved weights")

        # Collect voxels from all subjects
        all_voxels_list = []
        n_voxels_per_subject = {}
        subjects_included = []

        for subject in subjects:
            try:
                print(f"\n  Loading {subject}...")

                # Load weight matrix
                weights = self.load_weight_matrix(subject)

                # Extract feature block(s) and concatenate along feature axis
                feature_blocks = [self.extract_feature_block(weights, fs) for fs in feature_spaces]
                feature_weights = np.concatenate(feature_blocks, axis=0)

                # Select ROI voxels
                voxel_mask, voxel_indices_isc = self.select_voxels_roi_filtered(
                    subject, roi, roi_type=roi_type, min_performance=min_performance
                )

                n_voxels = len(voxel_indices_isc)

                if n_voxels == 0:
                    print(f"    WARNING: No voxels found for {subject}, skipping")
                    continue

                # Extract ROI weights: (n_features, n_voxels) -> (n_voxels, n_features)
                roi_weights = feature_weights[:, voxel_indices_isc].T

                print(f"    Loaded {n_voxels} voxels × {feature_weights.shape[0]} features")

                all_voxels_list.append(roi_weights)
                n_voxels_per_subject[subject] = n_voxels
                subjects_included.append(subject)

            except Exception as e:
                print(f"    ERROR loading {subject}: {e}")
                continue

        if len(all_voxels_list) == 0:
            raise ValueError("No subjects successfully loaded!")

        # Stack all voxels from all subjects
        pooled_voxels = np.vstack(all_voxels_list)  # (total_voxels, n_features)
        total_voxels, n_features = pooled_voxels.shape

        print(f"\n  Pooled voxels: {total_voxels} voxels × {n_features} features")
        print(f"  Subjects included: {len(subjects_included)}")
        print(f"  Voxels per subject: min={min(n_voxels_per_subject.values())}, "
              f"max={max(n_voxels_per_subject.values())}, "
              f"mean={np.mean(list(n_voxels_per_subject.values())):.1f}")

        # Within-voxel normalization (z-score each voxel across features)
        voxel_means = np.mean(pooled_voxels, axis=1, keepdims=True)
        voxel_stds = np.std(pooled_voxels, axis=1, keepdims=True)
        voxel_stds = np.where(voxel_stds == 0, 1.0, voxel_stds)  # Avoid division by zero
        pooled_voxels_norm = (pooled_voxels - voxel_means) / voxel_stds

        print(f"  Applied within-voxel normalization (z-score across features)")

        # Compute PCA on pooled data
        pca = PCA(n_components=n_components, whiten=whiten, svd_solver='auto')
        pca.fit(pooled_voxels_norm)

        n_components_actual = pca.n_components_
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)

        print(f"\n  PCA Results:")
        print(f"    Components retained: {n_components_actual}")
        print(f"    Cumulative variance: {cumulative_var[-1]:.4f}")
        print(f"    Top 5 PCs explain: {cumulative_var[min(4, n_components_actual-1)]:.4f}")

        # Package results
        pooled_pca_results = {
            'loadings': pca.components_,  # (K, D) - shared PC loadings
            'explained_var': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_var,
            'n_voxels_per_subject': n_voxels_per_subject,
            'total_voxels': total_voxels,
            'n_components': n_components_actual,
            'n_features': n_features,
            'subjects': subjects_included,
            'roi': roi,
            'roi_type': roi_type,
            'feature_space': feature_space_label,
            'feature_spaces': feature_spaces,
            'pca_model': pca,
            'voxel_means': voxel_means,
            'voxel_stds': voxel_stds
        }

        print(f"\n  Pooled PCA complete!")

        return pooled_pca_results

    def save_pooled_pca_results(self, pooled_pca_results):
        """
        Save pooled PCA results to disk.

        Saves to analysis/PooledPCA/ directory with structure:
        - loadings/{feature_space}_{roi_type}_{roi}_{model}_pooled_pc_loadings.h5
        - variance/{feature_space}_{roi_type}_{roi}_{model}_pooled_variance.csv
        - summary/{feature_space}_{roi_type}_{roi}_{model}_pooled_summary.json

        Args:
            pooled_pca_results: dict from compute_pooled_pca_across_subjects()
        """
        # Create output directory
        pooled_dir = os.path.join(os.path.dirname(self.out_dir), 'PooledPCA')
        os.makedirs(os.path.join(pooled_dir, 'loadings'), exist_ok=True)
        os.makedirs(os.path.join(pooled_dir, 'variance'), exist_ok=True)
        os.makedirs(os.path.join(pooled_dir, 'summary'), exist_ok=True)

        feature_space = pooled_pca_results['feature_space']
        feature_spaces = pooled_pca_results.get('feature_spaces', [feature_space])
        roi = pooled_pca_results['roi']
        roi_type = pooled_pca_results['roi_type']
        label = pooled_label(feature_space, roi_type, roi, self.model, self.population)

        # 1. Save PC loadings (HDF5) - main output for interpretation
        loadings_path = os.path.join(pooled_dir, 'loadings', f"{label}_pc_loadings.h5")
        with h5py.File(loadings_path, 'w') as f:
            f.create_dataset('loadings', data=pooled_pca_results['loadings'])
            f.create_dataset('explained_var', data=pooled_pca_results['explained_var'])
            f.create_dataset('cumulative_variance', data=pooled_pca_results['cumulative_variance'])

            # Metadata
            f.attrs['n_components'] = pooled_pca_results['n_components']
            f.attrs['n_features'] = pooled_pca_results['n_features']
            f.attrs['total_voxels'] = pooled_pca_results['total_voxels']
            f.attrs['n_subjects'] = len(pooled_pca_results['subjects'])
            f.attrs['feature_space'] = feature_space
            f.attrs['feature_spaces'] = ','.join(feature_spaces)
            f.attrs['roi'] = roi
            f.attrs['roi_type'] = roi_type
            f.attrs['population'] = self.population
            f.attrs['date'] = datetime.now().isoformat()

        print(f"  Saved loadings: {loadings_path}")

        # 2. Save variance explained (CSV)
        n_components = pooled_pca_results['n_components']
        variance_df = pd.DataFrame({
            'PC': np.arange(1, n_components + 1),
            'variance_explained': pooled_pca_results['explained_var'],
            'cumulative_variance': pooled_pca_results['cumulative_variance']
        })

        variance_path = os.path.join(pooled_dir, 'variance', f"{label}_variance.csv")
        variance_df.to_csv(variance_path, index=False)
        print(f"  Saved variance: {variance_path}")

        # 3. Save summary with per-subject voxel counts (JSON)
        import json

        # Convert numpy types to Python native types for JSON serialization
        n_voxels_per_subject_serializable = {
            str(k): int(v) for k, v in pooled_pca_results['n_voxels_per_subject'].items()
        }

        summary = {
            'feature_space': feature_space,
            'feature_spaces': feature_spaces,
            'roi': roi,
            'roi_type': roi_type,
            'n_subjects': len(pooled_pca_results['subjects']),
            'subjects': pooled_pca_results['subjects'],
            'n_voxels_per_subject': n_voxels_per_subject_serializable,
            'total_voxels': int(pooled_pca_results['total_voxels']),
            'n_components': int(pooled_pca_results['n_components']),
            'n_features': int(pooled_pca_results['n_features']),
            'cumulative_variance_top5': float(pooled_pca_results['cumulative_variance'][min(4, n_components-1)]),
            'cumulative_variance_all': float(pooled_pca_results['cumulative_variance'][-1]),
            'date': datetime.now().isoformat()
        }

        summary_path = os.path.join(pooled_dir, 'summary', f"{label}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary: {summary_path}")

    def _get_available_subjects(self):
        """Detect available subjects from weight files."""
        weights_dir = os.path.join(self.enc_dir, 'weights')
        weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.h5')]

        subjects = []
        for f in weight_files:
            if self.model in f:
                subject = f.split('_')[0]
                if subject not in subjects:
                    subjects.append(subject)

        subjects.sort()
        return subjects

    # ==================== Output Methods ====================

    def save_pca_results(self, pca_results, subject, feature_space, roi, roi_type, voxel_mask):
        """Save all PCA outputs with organized file structure."""
        label = f"{subject}_{feature_space}_{roi_type}_{roi}"

        # 1. Save PCA model (pickle)
        model_data = {
            'pca_model': pca_results['pca_model'],
            'voxel_means': pca_results['voxel_means'],
            'voxel_stds': pca_results['voxel_stds'],
            'metadata': {
                'subject': subject,
                'feature_space': feature_space,
                'roi': roi,
                'roi_type': roi_type,
                'n_voxels': pca_results['pc_scores'].shape[0],
                'n_features': pca_results['pc_loadings'].shape[1],
                'n_components': pca_results['n_components'],
                'date': datetime.now().isoformat()
            }
        }

        model_path = os.path.join(self.out_dir, 'models', f"{label}_pca_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # 2. Save PC scores (HDF5)
        scores_path = os.path.join(self.out_dir, 'scores', f"{label}_pc_scores.h5")
        with h5py.File(scores_path, 'w') as f:
            f.create_dataset('pc_scores', data=pca_results['pc_scores'])
            f.create_dataset('variance_explained', data=pca_results['variance_explained'])
            f.attrs['n_components'] = pca_results['n_components']

        # 3. Save PC scores as brain maps (NIfTI)
        brain_map_path = os.path.join(self.out_dir, 'scores', f"{label}_pc_scores.nii.gz")
        self._save_brain_maps(pca_results['pc_scores'], voxel_mask, brain_map_path)

        # 4. Save PC loadings (CSV)
        n_components = pca_results['n_components']
        loadings_df = pd.DataFrame(
            pca_results['pc_loadings'].T,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        loadings_df.insert(0, 'feature_dim', [f'dim_{i}' for i in range(len(loadings_df))])

        loadings_path = os.path.join(self.out_dir, 'loadings', f"{label}_pc_loadings.csv")
        loadings_df.to_csv(loadings_path, index=False)

        # 5. Save variance explained (CSV)
        variance_df = pd.DataFrame({
            'PC': np.arange(1, n_components + 1),
            'variance_explained': pca_results['variance_explained'],
            'cumulative_variance': pca_results['cumulative_variance']
        })

        variance_path = os.path.join(self.out_dir, 'variance', f"{label}_variance.csv")
        variance_df.to_csv(variance_path, index=False)

    def _save_brain_maps(self, pc_scores, voxel_mask, output_path):
        """Map PC scores to full brain volume and save as 4D NIfTI."""
        n_components = pc_scores.shape[1]

        # Create 4D volume: (x, y, z, n_components)
        brain_maps = np.zeros(self.brain_shape + (n_components,))
        brain_maps[:] = np.nan

        # Fill in voxel scores
        for i in range(n_components):
            brain_maps[..., i][voxel_mask] = pc_scores[:, i]

        # Save as NIfTI
        img = nibabel.Nifti1Image(brain_maps, affine=self.affine)
        nibabel.save(img, output_path)

    # ==================== Visualization ====================

    def _generate_visualizations(self, pca_results, subject, feature_space, roi, roi_type):
        """Generate all visualizations for one analysis."""
        label = f"{subject}_{feature_space}_{roi_type}_{roi}"
        fig_dir = os.path.join(self.out_dir, 'figures')

        # 1. Variance explained plot
        self.plot_variance_explained(
            pca_results['variance_explained'],
            pca_results['cumulative_variance'],
            feature_space, roi, roi_type,
            os.path.join(fig_dir, 'variance_explained', f"{label}_variance.png")
        )

        # 2. Feature loadings for top 3 PCs
        n_pcs_to_plot = min(3, pca_results['n_components'])
        for pc_index in range(n_pcs_to_plot):
            self.plot_feature_loadings(
                pca_results['pc_loadings'],
                feature_space, pc_index,
                os.path.join(fig_dir, 'feature_loadings', f"{label}_PC{pc_index+1}_loadings.png")
            )

    def plot_variance_explained(self, variance_ratios, cumulative_variance,
                               feature_space, roi, roi_type, output_path):
        """Create scree plot showing variance explained by each PC."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Individual variance
        axes[0].bar(range(1, len(variance_ratios) + 1), variance_ratios,
                   color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Variance Explained')
        axes[0].set_title(f'{feature_space} - {roi} ({roi_type})\nVariance per PC')
        axes[0].grid(alpha=0.3)

        # Panel 2: Cumulative variance
        axes[1].plot(range(1, len(cumulative_variance) + 1),
                    cumulative_variance, 'o-', color='darkred', linewidth=2)
        axes[1].axhline(0.90, color='gray', linestyle='--', label='90% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Variance Explained')
        axes[1].set_title('Cumulative Variance')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_loadings(self, loadings, feature_space, pc_index, output_path, top_k=20):
        """Visualize top feature loadings for a specific PC."""
        pc_loadings = loadings[pc_index, :]

        # Get top K features by absolute loading
        top_indices = np.argsort(np.abs(pc_loadings))[-top_k:][::-1]
        top_loadings = pc_loadings[top_indices]
        top_labels = [f'dim_{i}' for i in top_indices]

        # Color by sign
        colors = ['crimson' if x < 0 else 'steelblue' for x in top_loadings]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 10))
        y_pos = np.arange(len(top_loadings))
        ax.barh(y_pos, top_loadings, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_labels)
        ax.set_xlabel('PC Loading')
        ax.set_title(f'{feature_space} - PC{pc_index+1}\nTop {top_k} Features')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pc_interpretation_results(
        self,
        results_csv_path=None,
        roi_types=['func', 'anat'],
        rois=None,
        output_dir=None,
        figsize_per_roi=(3.5, 5)
    ):
        """
        Plot PC interpretation results (feature prediction from PC time series).

        Creates separate barplots for functional and anatomical ROIs showing
        mean R² values across subjects with individual subject points overlaid.

        Args:
            results_csv_path: Path to all_pc_interpretation_results.csv
                             If None, defaults to analysis/PCInterpretation/
            roi_types: List of ROI types ['func', 'anat']
            rois: List of ROIs (default: ['MT', 'STS', 'temporal_language', 'frontal'])
            output_dir: Output directory (default: out_dir/figures/pc_interpretation/)
            figsize_per_roi: Size per subplot (width, height)

        Returns:
            dict: {roi_type: {'bar': fig, 'combined': fig}} for each roi_type
        """
        # Set default paths
        if results_csv_path is None:
            results_csv_path = os.path.join(
                self.dir, 'analysis', 'PCInterpretation',
                'all_pc_interpretation_results.csv'
            )

        if rois is None:
            rois = ['MT', 'STS', 'temporal_language', 'frontal']

        if output_dir is None:
            output_dir = os.path.join(self.dir, 'figures', 'subfigures', 'WeightPCA')

        # Load and validate data
        if not os.path.exists(results_csv_path):
            raise FileNotFoundError(f"PC interpretation results not found: {results_csv_path}")

        print(f"\nLoading PC interpretation results from: {results_csv_path}")
        df = pd.read_csv(results_csv_path)

        # Validate required columns
        required_cols = ['subject', 'roi', 'roi_type', 'label', 'r2_mean']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        print(f"Loaded {len(df)} rows of data")

        # Get colors
        colors_dict = plotting_helpers.get_colors_dict()

        # Define feature order (grouped semantically)
        feature_order = [
            'social', 'num_agents', 'turn_taking', 'mentalization',  # Social/cognitive
            'valence', 'arousal',                                     # Emotion
            'motion', 'face', 'hue', 'pixel',                        # Visual
            'speaking', 'music', 'amplitude', 'pitch',               # Audio
            'written_text'                                            # Language
        ]

        # Define network families and their layers
        network_families = {
            'alexnet': ['alexnet_layer2','alexnet_layer5','alexnet_layer6', 'alexnet_layer7'],
            'GPT2': ['GPT2_chunk60_layer2', 'GPT2_chunk60_layer3', 'GPT2_chunk60_layer4',
                     'GPT2_chunk60_layer10', 'GPT2_chunk60_layer11', 'GPT2_chunk60_layer12'],
            'hubert': ['hubert_chunk60_layer2', 'hubert_chunk60_layer3', 'hubert_chunk60_layer4',
                      'hubert_chunk60_layer10', 'hubert_chunk60_layer11', 'hubert_chunk60_layer12']
        }
        

        import seaborn as sns
        from matplotlib.patches import Patch
        import math

        figures = {}

        # Iterate over each combination of ROI type, ROI, and network
        for roi_type in roi_types:
            print(f"\nProcessing roi_type: {roi_type}")

            # Filter data for this roi_type
            df_type = df[df['roi_type'] == roi_type].copy()

            if len(df_type) == 0:
                print(f"WARNING: No data for roi_type={roi_type}, skipping")
                continue

            # Prepare data for plotting (per-subject data)
            # Use correlation if available, otherwise fall back to r2_mean
            if 'correlation' in df_type.columns:
                print("  Using correlation values (taking absolute value)")
                plot_df = df_type[['subject', 'roi', 'feature_space', 'label', 'correlation']].copy()
                plot_df['abs_correlation'] = plot_df['correlation'].abs()
                value_col = 'abs_correlation'
                ylabel = '|r| (absolute correlation)'
            else:
                print("  Using R² values")
                plot_df = df_type[['subject', 'roi', 'feature_space', 'label', 'r2_mean']].copy()
                plot_df = plot_df.rename(columns={'r2_mean': 'r2'})
                value_col = 'r2'
                ylabel = 'R² (feature prediction)'

            # Filter to available ROIs
            roi_order = [r for r in rois if r in plot_df['roi'].unique()]

            print(f"  ROIs: {roi_order}")
            print(f"  Feature spaces: {plot_df['feature_space'].unique()}")

            # Create a plot for each ROI × network combination
            for roi in roi_order:
                roi_data = plot_df[plot_df['roi'] == roi].copy()

                for network, layers in network_families.items():
                    # Filter to this network's layers
                    network_data = roi_data[roi_data['feature_space'].isin(layers)].copy()

                    if len(network_data) == 0:
                        print(f"  WARNING: No data for {roi} ({roi_type}) - {network}, skipping")
                        continue

                    available_layers = [l for l in layers if l in network_data['feature_space'].unique()]
                    n_layers = len(available_layers)

                    if n_layers == 0:
                        continue

                    print(f"  Creating plot: {roi} ({roi_type}) - {network} ({n_layers} layers)")

                    # Determine grid layout (prefer more columns than rows)
                    ncols = min(3, n_layers)  # Max 3 columns
                    nrows = math.ceil(n_layers / ncols)

                    # Create figure with subplots
                    fig_width = figsize_per_roi[0] * ncols
                    fig_height = figsize_per_roi[1] * nrows

                    bar_fig, bar_axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                                                     squeeze=False, sharey=True)
                    strip_fig, strip_axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                                                         squeeze=False, sharey=True)

                    # Plot each layer in its own subplot
                    for idx, layer in enumerate(available_layers):
                        row = idx // ncols
                        col = idx % ncols

                        layer_data = network_data[network_data['feature_space'] == layer].copy()

                        # Plot bar only version
                        sns.barplot(
                            data=layer_data,
                            x='feature_space',
                            y=value_col,
                            hue='label',
                            order=[layer],
                            hue_order=feature_order,
                            palette=colors_dict,
                            errorbar='se',
                            ax=bar_axes[row, col],
                            edgecolor="black",
                            linewidth=1.5
                        )

                        # Plot combined version (bars + strips)
                        sns.barplot(
                            data=layer_data,
                            x='feature_space',
                            y=value_col,
                            hue='label',
                            order=[layer],
                            hue_order=feature_order,
                            palette=colors_dict,
                            errorbar='se',
                            ax=strip_axes[row, col],
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=1.5
                        )
                        sns.stripplot(
                            data=layer_data,
                            x='feature_space',
                            y=value_col,
                            hue='label',
                            order=[layer],
                            hue_order=feature_order,
                            palette=colors_dict,
                            ax=strip_axes[row, col],
                            dodge=True,
                            size=3,
                            alpha=0.6,
                            jitter=0.2,
                            edgecolor="black",
                            linewidth=0.5
                        )

                        # Style each subplot
                        for ax in [bar_axes[row, col], strip_axes[row, col]]:
                            # Remove legend
                            if ax.get_legend():
                                ax.get_legend().remove()

                            # Clean up axis labels
                            ax.set_xlabel('')
                            ax.set_xticklabels([])
                            ax.set_ylabel(ylabel if col == 0 else '')
                            ax.axhline(0, color='black', linewidth=1, alpha=0.5)
                            ax.set_title(layer.replace('_chunk60', ''), fontsize=12)

                    # Hide unused subplots
                    for idx in range(n_layers, nrows * ncols):
                        row = idx // ncols
                        col = idx % ncols
                        bar_axes[row, col].set_visible(False)
                        strip_axes[row, col].set_visible(False)

                    # Add overall title
                    bar_fig.suptitle(f'PC Interpretation: {roi} ({roi_type}) - {network}',
                                    fontsize=16, fontweight='normal')
                    strip_fig.suptitle(f'PC Interpretation: {roi} ({roi_type}) - {network}',
                                      fontsize=16, fontweight='normal')

                    # Create legend
                    legend_elements = [Patch(facecolor=colors_dict.get(feat, 'gray'), label=feat)
                                      for feat in feature_order if feat in network_data['label'].unique()]

                    # Add legend below
                    bar_fig.legend(handles=legend_elements, loc='lower center',
                                  ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10)
                    strip_fig.legend(handles=legend_elements, loc='lower center',
                                    ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10)

                    # Tight layout
                    plt.figure(bar_fig.number)
                    plt.tight_layout()
                    plt.figure(strip_fig.number)
                    plt.tight_layout()

                    # Save figures
                    Path(output_dir).mkdir(exist_ok=True, parents=True)

                    filename_base = f'pc_interpretation_{roi_type}_{roi}_{network}'

                    output_bar = os.path.join(output_dir, f'{filename_base}_bar.png')
                    bar_fig.savefig(output_bar, dpi=300, bbox_inches='tight')
                    print(f"    Saved: {output_bar}")

                    output_combined = os.path.join(output_dir, f'{filename_base}_combined.png')
                    strip_fig.savefig(output_combined, dpi=300, bbox_inches='tight')
                    print(f"    Saved: {output_combined}")

                    # Store figures
                    key = f"{roi_type}_{roi}_{network}"
                    figures[key] = {'bar': bar_fig, 'combined': strip_fig}

                    # Close to free memory
                    plt.close(bar_fig)
                    plt.close(strip_fig)

        print(f"\nPC interpretation plotting complete!")
        print(f"Output directory: {output_dir}")

        return figures

    def plot_pc_interpretation_per_pc_results(
        self,
        results_csv_path=None,
        roi_types=['func', 'anat'],
        rois=None,
        output_dir=None,
        figsize_per_cell=(3.5, 5),
        pcs=[0, 1, 2]
    ):
        """
        Plot per-PC interpretation results (feature prediction from individual PC time series).

        Creates separate barplots for functional and anatomical ROIs showing
        mean R² values across subjects with individual subject points overlaid.
        Each network family plot has one row per layer and one column per PC.

        Args:
            results_csv_path: Path to per-PC interpretation results CSV
                             If None, defaults to analysis/PCInterpretation/
            roi_types: List of ROI types ['func', 'anat']
            rois: List of ROIs (default: ['MT', 'STS', 'temporal_language', 'frontal'])
            output_dir: Output directory (default: figures/subfigures/WeightPCA/)
            figsize_per_cell: Size per subplot (width, height)
            pcs: List of PC indices to plot (default: [0, 1, 2])

        Returns:
            dict: {roi_type_roi_network: {'bar': fig, 'combined': fig}} for each combination
        """
        # Set default paths
        if results_csv_path is None:
            results_csv_path = os.path.join(
                self.dir, 'analysis', 'PCInterpretation',
                'all_per_pc_results.csv'
            )

        if rois is None:
            rois = ['MT', 'STS', 'temporal_language', 'frontal']

        if output_dir is None:
            output_dir = os.path.join(self.dir, 'figures', 'subfigures', 'WeightPCA')

        # Load and validate data
        if not os.path.exists(results_csv_path):
            raise FileNotFoundError(f"Per-PC interpretation results not found: {results_csv_path}")

        print(f"\nLoading per-PC interpretation results from: {results_csv_path}")
        df = pd.read_csv(results_csv_path)

        # Validate required columns
        required_cols = ['subject', 'roi', 'roi_type', 'label', 'r2_mean', 'pc_index']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        print(f"Loaded {len(df)} rows of data")

        # Get colors
        colors_dict = plotting_helpers.get_colors_dict()

        # Define feature order (grouped semantically)
        feature_order = [
            'social', 'num_agents', 'turn_taking', 'mentalization',  # Social/cognitive
            'valence', 'arousal',                                     # Emotion
            'motion', 'face', 'hue', 'pixel',                        # Visual
            'speaking', 'music', 'amplitude', 'pitch',               # Audio
            'written_text'                                            # Language
        ]

        # Define network families and their layers
        network_families = {
            'alexnet': ['alexnet_layer2','alexnet_layer5','alexnet_layer6', 'alexnet_layer7'],
            'GPT2': ['GPT2_chunk60_layer2', 'GPT2_chunk60_layer3', 'GPT2_chunk60_layer4',
                     'GPT2_chunk60_layer10', 'GPT2_chunk60_layer11', 'GPT2_chunk60_layer12'],
            'hubert': ['hubert_chunk60_layer2', 'hubert_chunk60_layer3', 'hubert_chunk60_layer4',
                      'hubert_chunk60_layer10', 'hubert_chunk60_layer11', 'hubert_chunk60_layer12']
        }

        import seaborn as sns
        from matplotlib.patches import Patch
        import math

        figures = {}

        # Iterate over each combination of ROI type, ROI, and network
        for roi_type in roi_types:
            print(f"\nProcessing roi_type: {roi_type}")

            # Filter data for this roi_type
            df_type = df[df['roi_type'] == roi_type].copy()

            if len(df_type) == 0:
                print(f"WARNING: No data for roi_type={roi_type}, skipping")
                continue

            # Prepare data for plotting (per-subject data)
            # Use correlation if available, otherwise fall back to r2_mean
            # if 'correlation' in df_type.columns:
            #     print("  Using correlation values (taking absolute value)")
            #     plot_df = df_type[['subject', 'roi', 'feature_space', 'label', 'pc_index', 'correlation']].copy()
            #     plot_df['abs_correlation'] = plot_df['correlation'].abs()
            #     value_col = 'abs_correlation'
            #     ylabel = '|r|'
            # else:
            print("  Using R² values")
            plot_df = df_type[['subject', 'roi', 'feature_space', 'label', 'pc_index', 'r2_mean']].copy()
            plot_df = plot_df.rename(columns={'r2_mean': 'r2'})
            value_col = 'r2'
            ylabel = 'r²'

            # Filter to available ROIs
            roi_order = [r for r in rois if r in plot_df['roi'].unique()]

            print(f"  ROIs: {roi_order}")
            print(f"  Feature spaces: {plot_df['feature_space'].unique()}")

            # Create a plot for each ROI × network combination
            for roi in roi_order:
                roi_data = plot_df[plot_df['roi'] == roi].copy()

                for network, layers in network_families.items():
                    # Filter to this network's layers
                    network_data = roi_data[roi_data['feature_space'].isin(layers)].copy()

                    if len(network_data) == 0:
                        print(f"  WARNING: No data for {roi} ({roi_type}) - {network}, skipping")
                        continue

                    available_layers = [l for l in layers if l in network_data['feature_space'].unique()]
                    n_layers = len(available_layers)

                    if n_layers == 0:
                        continue

                    print(f"  Creating per-PC plot: {roi} ({roi_type}) - {network} ({n_layers} layers x {len(pcs)} PCs)")

                    # Grid layout: rows = layers, cols = PCs
                    nrows = n_layers
                    ncols = len(pcs)

                    # Create figure with subplots
                    fig_width = figsize_per_cell[0] * ncols
                    fig_height = figsize_per_cell[1] * nrows

                    bar_fig, bar_axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                                                     squeeze=False, sharey=True)
                    strip_fig, strip_axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                                                         squeeze=False, sharey=True)

                    # Plot each layer (row) × PC (column) combination
                    for layer_idx, layer in enumerate(available_layers):
                        layer_data = network_data[network_data['feature_space'] == layer].copy()

                        for pc_col_idx, pc_index in enumerate(pcs):
                            # Filter to this PC
                            pc_data = layer_data[layer_data['pc_index'] == pc_index].copy()

                            if len(pc_data) == 0:
                                # No data for this PC, leave subplot empty
                                bar_axes[layer_idx, pc_col_idx].text(0.5, 0.5, 'No data',
                                                                     ha='center', va='center',
                                                                     transform=bar_axes[layer_idx, pc_col_idx].transAxes)
                                strip_axes[layer_idx, pc_col_idx].text(0.5, 0.5, 'No data',
                                                                       ha='center', va='center',
                                                                       transform=strip_axes[layer_idx, pc_col_idx].transAxes)
                                continue

                            # Plot bar only version
                            sns.barplot(
                                data=pc_data,
                                x='feature_space',
                                y=value_col,
                                hue='label',
                                order=[layer],
                                hue_order=feature_order,
                                palette=colors_dict,
                                errorbar='se',
                                ax=bar_axes[layer_idx, pc_col_idx],
                                edgecolor="black",
                                linewidth=1.5
                            )

                            # Plot combined version (bars + strips)
                            sns.barplot(
                                data=pc_data,
                                x='feature_space',
                                y=value_col,
                                hue='label',
                                order=[layer],
                                hue_order=feature_order,
                                palette=colors_dict,
                                errorbar='se',
                                ax=strip_axes[layer_idx, pc_col_idx],
                                alpha=0.7,
                                edgecolor="black",
                                linewidth=1.5
                            )
                            sns.stripplot(
                                data=pc_data,
                                x='feature_space',
                                y=value_col,
                                hue='label',
                                order=[layer],
                                hue_order=feature_order,
                                palette=colors_dict,
                                ax=strip_axes[layer_idx, pc_col_idx],
                                dodge=True,
                                size=3,
                                alpha=0.6,
                                jitter=0.2,
                                edgecolor="black",
                                linewidth=0.5
                            )

                            # Style each subplot
                            for ax in [bar_axes[layer_idx, pc_col_idx], strip_axes[layer_idx, pc_col_idx]]:
                                # Remove legend
                                if ax.get_legend():
                                    ax.get_legend().remove()

                                # Clean up axis labels
                                ax.set_xlabel('')
                                ax.set_xticklabels([])

                                # Y-label only on leftmost column
                                if pc_col_idx == 0:
                                    ax.set_ylabel(f'{layer.replace("_chunk60", "")}\n{ylabel}', fontsize=10)
                                else:
                                    ax.set_ylabel('')

                                ax.axhline(0, color='black', linewidth=1, alpha=0.5)

                                # Title only on top row
                                if layer_idx == 0:
                                    ax.set_title(f'PC {pc_index}', fontsize=12)

                    # Add overall title
                    bar_fig.suptitle(f'Per-PC Interpretation: {roi} ({roi_type}) - {network}',
                                    fontsize=16, fontweight='normal')
                    strip_fig.suptitle(f'Per-PC Interpretation: {roi} ({roi_type}) - {network}',
                                      fontsize=16, fontweight='normal')

                    # Create legend
                    legend_elements = [Patch(facecolor=colors_dict.get(feat, 'gray'), label=feat)
                                      for feat in feature_order if feat in network_data['label'].unique()]

                    # Add legend below
                    bar_fig.legend(handles=legend_elements, loc='lower center',
                                  ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10)
                    strip_fig.legend(handles=legend_elements, loc='lower center',
                                    ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10)

                    # Tight layout
                    plt.figure(bar_fig.number)
                    plt.tight_layout()
                    plt.figure(strip_fig.number)
                    plt.tight_layout()

                    # Save figures
                    Path(output_dir).mkdir(exist_ok=True, parents=True)

                    filename_base = f'per_pc_interpretation_{roi_type}_{roi}_{network}'

                    output_bar = os.path.join(output_dir, f'{filename_base}_bar.png')
                    bar_fig.savefig(output_bar, dpi=300, bbox_inches='tight')
                    print(f"    Saved: {output_bar}")

                    output_combined = os.path.join(output_dir, f'{filename_base}_combined.png')
                    strip_fig.savefig(output_combined, dpi=300, bbox_inches='tight')
                    print(f"    Saved: {output_combined}")

                    # Store figures
                    key = f"{roi_type}_{roi}_{network}"
                    figures[key] = {'bar': bar_fig, 'combined': strip_fig}

                    # Close to free memory
                    plt.close(bar_fig)
                    plt.close(strip_fig)

        print(f"\nPer-PC interpretation plotting complete!")
        print(f"Output directory: {output_dir}")

        return figures

    def plot_pooled_pca_results(
        self,
        results_csv_path=None,
        roi_types=['func', 'anat'],
        rois=None,
        output_dir=None,
        figsize_per_cell=(3.5, 5),
        use_abs_correlation=False,
        pcs=[0, 1, 2, 3,4],
        uniform_ylim=True
    ):
        """
        Plot pooled PCA interpretation results (correlations between pooled PC time series and features).

        Creates separate barplots for functional and anatomical ROIs showing
        mean absolute correlation values across subjects with individual subject points overlaid.
        Each network family plot has one row per layer and one column per PC.

        Args:
            results_csv_path: Path to pooled PCA interpretation results CSV
                             If None, defaults to analysis/PCInterpretation/
            roi_types: List of ROI types ['func', 'anat']
            rois: List of ROIs (default: ['MT', 'STS', 'temporal_language', 'frontal'])
            output_dir: Output directory (default: figures/subfigures/WeightPCA/)
            figsize_per_cell: Size per subplot (width, height)
            use_abs_correlation: If True, plot absolute value of correlation (default: True)
                                Sign of correlation is arbitrary due to PC orientation
            pcs: List of PC indices to plot (default: [0, 1, 2])
            uniform_ylim: If True, use the same y-axis limits across all subplots (default: True)

        Returns:
            dict: {roi_type_roi_network: {'bar': fig, 'combined': fig}} for each combination
        """
        # Set default paths
        if results_csv_path is None:
            results_csv_path = os.path.join(
                self.dir, 'analysis', 'PCInterpretation',
                'compiled_pooled_results.csv'
            )

        if rois is None:
            rois = ['MT', 'STS', 'temporal_language', 'frontal']

        if output_dir is None:
            output_dir = os.path.join(self.dir, 'figures', 'subfigures', 'WeightPCA')

        # Load and validate data
        if not os.path.exists(results_csv_path):
            raise FileNotFoundError(f"Pooled PCA results not found: {results_csv_path}")

        print(f"\nLoading pooled PCA results from: {results_csv_path}")
        df = pd.read_csv(results_csv_path)

        # Validate required columns
        required_cols = ['subject', 'roi', 'roi_type', 'label', 'correlation', 'pc_index']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        print(f"Loaded {len(df)} rows of data")

        # Take absolute value of correlation if requested
        # (PC sign is arbitrary, so |correlation| is more interpretable)
        if use_abs_correlation:
            print("Using absolute value of correlations (PC sign is arbitrary)")
            df['abs_correlation'] = df['correlation'].abs()
            value_col = 'abs_correlation'
            ylabel = '|r|'
        else:
            print("  Using R² values")
            # plot_df = df_type[['subject', 'roi', 'feature_space', 'label', 'pc_index', 'r2_mean']].copy()
            # plot_df = plot_df.rename(columns={'r2_mean': 'r2'})
            value_col = 'r2_mean'
            ylabel = 'r²'
            # value_col = 'correlation'
            # ylabel = 'r'

        # Detect significance / null columns (graceful fallback for old CSVs).
        # Only meaningful when plotting r² (the permutation null is on r²).
        has_sig = (value_col == 'r2_mean'
                   and all(c in df.columns for c in ['q_value_fdr', 'null_r2_95']))
        if has_sig:
            print("  Found q_value_fdr + null_r2_95 — will overlay significance markers")
        else:
            print("  Skipping significance overlays (columns missing or plotting correlation)")

        # Get colors
        colors_dict = plotting_helpers.get_colors_dict()

        # Define feature order (grouped semantically)
        feature_order = [
            'motion','pixel','hue','face','num_agents', 'written_text','amplitude','pitch',
            'valence', 'arousal', 'social', 'turn_taking', 'mentalization','speaking'  # Social/cognitive
        ]

        # Define network families and their layers
        network_families = {
            'alexnet': ['alexnet_layer2','alexnet_layer5','alexnet_layer6', 'alexnet_layer7'],
            'GPT2': ['GPT2_chunk60_layer2', 'GPT2_chunk60_layer3', 'GPT2_chunk60_layer4',
                     'GPT2_chunk60_layer10', 'GPT2_chunk60_layer11', 'GPT2_chunk60_layer12'],
            'hubert': ['hubert_chunk60_layer2', 'hubert_chunk60_layer3', 'hubert_chunk60_layer4',
                      'hubert_chunk60_layer10', 'hubert_chunk60_layer11', 'hubert_chunk60_layer12']
        }
        
        network_families = {
            'alexnet': ['alexnet_layer6'],
            'GPT2': ['GPT2_chunk60_layer11'],
            'hubert': ['hubert_chunk60_layer2', 'hubert_chunk60_layer3', 'hubert_chunk60_layer4',
                      'hubert_chunk60_layer10', 'hubert_chunk60_layer11', 'hubert_chunk60_layer12']
        }
        network_families = {
            'alexnet': ['alexnet_eps0.3_layer5+alexnet_eps0.3_layer6+alexnet_eps0.3_layer7'],
            'GPT2': ['GPT2_chunk120_layer10+GPT2_chunk120_layer11+GPT2_chunk120_layer12'],
            'hubert': ['hubert_chunk120_layer10+hubert_chunk120_layer11+hubert_chunk120_layer12'],
            'motion': ['motion']
        }

        import seaborn as sns
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        import math

        def _overlay_significance(ax, pc_data, layer, hue_order, value_col):
            """Add FDR-asterisks + null-95 dashed lines on each dodged bar in `ax`."""
            # Collapse subjects → one row per label (sig columns are constant within)
            sig_row = (pc_data.groupby('label', as_index=False)
                              .agg({'q_value_fdr': 'first', 'null_r2_95': 'first', value_col: 'mean'}))
            sig_row = sig_row.set_index('label')

            # Bars from seaborn are Rectangle patches, one per hue level (order=[layer])
            bar_patches = [p for p in ax.patches
                           if hasattr(p, 'get_height') and p.get_width() > 0]
            for hue_idx, hue_val in enumerate(hue_order):
                if hue_idx >= len(bar_patches) or hue_val not in sig_row.index:
                    continue
                patch = bar_patches[hue_idx]
                bar_x = patch.get_x() + patch.get_width() / 2
                bar_w = patch.get_width()
                bar_top = patch.get_height()
                # Skip bars with non-finite geometry (e.g. all-NaN label group)
                if not (np.isfinite(bar_x) and np.isfinite(bar_top) and np.isfinite(bar_w)):
                    continue

                q = sig_row.loc[hue_val, 'q_value_fdr']
                null95 = sig_row.loc[hue_val, 'null_r2_95']

                # Null 95th percentile reference line, per bar
                if pd.notna(null95):
                    ax.hlines(null95, bar_x - bar_w / 2, bar_x + bar_w / 2,
                              linestyles='--', colors='black', alpha=0.5, linewidth=0.8)

                # Tiered FDR asterisks above the bar
                if pd.notna(q):
                    if q < 0.001:
                        marker = '***'
                    elif q < 0.01:
                        marker = '**'
                    elif q < 0.05:
                        marker = '*'
                    else:
                        marker = ''
                    if marker:
                        ax.text(bar_x, max(bar_top, 0) + 0.005, marker,
                                ha='center', va='bottom', fontsize=8)

        figures = {}

        # Calculate per-feature y-axis limits from aggregated bar heights (mean ± SE),
        # restricted to the rois/labels/pcs/feature_spaces we actually plot.
        feature_ylims = {}
        if uniform_ylim:
            plotted_layers = [l for layers in network_families.values() for l in layers]
            ylim_df = df[
                df['roi_type'].isin(roi_types)
                & df['roi'].isin(rois)
                & df['label'].isin(feature_order)
                & df['pc_index'].isin(pcs)
                & df['feature_space'].isin(plotted_layers)
            ]
            agg = (ylim_df.groupby(['feature_space', 'roi_type', 'roi', 'label', 'pc_index'])[value_col]
                          .agg(['mean', 'sem'])
                          .reset_index())
            agg['hi'] = agg['mean'] + agg['sem'].fillna(0)
            agg['lo'] = agg['mean'] - agg['sem'].fillna(0)
            # Also ensure the null-95 reference line stays visible if present
            if has_sig:
                null95_by_key = (ylim_df
                                 .groupby(['feature_space', 'roi_type', 'roi', 'label', 'pc_index'])
                                 ['null_r2_95'].first().reset_index())
                agg = agg.merge(null95_by_key,
                                on=['feature_space', 'roi_type', 'roi', 'label', 'pc_index'],
                                how='left')
                agg['hi'] = agg[['hi', 'null_r2_95']].max(axis=1)
            for feat, feat_df in agg.groupby('feature_space'):
                f_min = min(feat_df['lo'].min(), 0)
                f_max = feat_df['hi'].max()
                f_range = f_max - f_min
                f_min = max(-1, f_min - 0.05 * f_range)
                f_max = f_max + 0.05 * f_range
                feature_ylims[feat] = (f_min, f_max)
                print(f"  ylim[{feat}] = [{f_min:.4f}, {f_max:.4f}]")

        # Iterate over each combination of ROI type, ROI, and network
        for roi_type in roi_types:
            print(f"\nProcessing roi_type: {roi_type}")

            # Filter data for this roi_type
            df_type = df[df['roi_type'] == roi_type].copy()

            if len(df_type) == 0:
                print(f"WARNING: No data for roi_type={roi_type}, skipping")
                continue

            # Prepare data for plotting (per-subject, per-PC data)
            plot_cols = ['subject', 'roi', 'feature_space', 'label', 'pc_index', value_col]
            if has_sig:
                plot_cols += ['q_value_fdr', 'null_r2_95']
            plot_df = df_type[plot_cols].copy()

            # Filter to available ROIs
            roi_order = [r for r in rois if r in plot_df['roi'].unique()]

            print(f"  ROIs: {roi_order}")
            print(f"  Feature spaces: {plot_df['feature_space'].unique()}")

            # Create a plot for each ROI × network combination
            for roi in roi_order:
                roi_data = plot_df[plot_df['roi'] == roi].copy()

                for network, layers in network_families.items():
                    # Filter to this network's layers
                    network_data = roi_data[roi_data['feature_space'].isin(layers)].copy()

                    if len(network_data) == 0:
                        print(f"  WARNING: No data for {roi} ({roi_type}) - {network}, skipping")
                        continue

                    available_layers = [l for l in layers if l in network_data['feature_space'].unique()]
                    n_layers = len(available_layers)

                    if n_layers == 0:
                        continue

                    print(f"  Creating per-PC plot: {roi} ({roi_type}) - {network} ({n_layers} layers x {len(pcs)} PCs)")

                    # Grid layout: rows = layers, cols = PCs
                    nrows = n_layers
                    ncols = len(pcs)

                    # Create figure with subplots
                    fig_width = figsize_per_cell[0] * ncols
                    fig_height = figsize_per_cell[1] * nrows

                    bar_fig, bar_axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                                                     squeeze=False, sharey=True)
                    strip_fig, strip_axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                                                         squeeze=False, sharey=True)

                    # Plot each layer (row) × PC (column) combination
                    for layer_idx, layer in enumerate(available_layers):
                        layer_data = network_data[network_data['feature_space'] == layer].copy()

                        for pc_col_idx, pc_index in enumerate(pcs):
                            # Filter to this PC
                            pc_data = layer_data[layer_data['pc_index'] == pc_index].copy()

                            if len(pc_data) == 0:
                                # No data for this PC, leave subplot empty
                                bar_axes[layer_idx, pc_col_idx].text(0.5, 0.5, 'No data',
                                                                     ha='center', va='center',
                                                                     transform=bar_axes[layer_idx, pc_col_idx].transAxes)
                                strip_axes[layer_idx, pc_col_idx].text(0.5, 0.5, 'No data',
                                                                       ha='center', va='center',
                                                                       transform=strip_axes[layer_idx, pc_col_idx].transAxes)
                                continue

                            # Plot bar only version
                            sns.barplot(
                                data=pc_data,
                                x='feature_space',
                                y=value_col,
                                hue='label',
                                order=[layer],
                                hue_order=feature_order,
                                palette=colors_dict,
                                errorbar='se',
                                ax=bar_axes[layer_idx, pc_col_idx],
                                edgecolor="black",
                                linewidth=1.5
                            )

                            # Plot combined version (bars + strips)
                            sns.barplot(
                                data=pc_data,
                                x='feature_space',
                                y=value_col,
                                hue='label',
                                order=[layer],
                                hue_order=feature_order,
                                palette=colors_dict,
                                errorbar='se',
                                ax=strip_axes[layer_idx, pc_col_idx],
                                alpha=0.7,
                                edgecolor="black",
                                linewidth=1.5
                            )
                            sns.stripplot(
                                data=pc_data,
                                x='feature_space',
                                y=value_col,
                                hue='label',
                                order=[layer],
                                hue_order=feature_order,
                                palette=colors_dict,
                                ax=strip_axes[layer_idx, pc_col_idx],
                                dodge=True,
                                size=3,
                                alpha=0.6,
                                jitter=0.2,
                                edgecolor="black",
                                linewidth=0.5
                            )

                            # Style each subplot
                            for ax in [bar_axes[layer_idx, pc_col_idx], strip_axes[layer_idx, pc_col_idx]]:
                                # Remove legend
                                if ax.get_legend():
                                    ax.get_legend().remove()

                                # Clean up axis labels
                                ax.set_xlabel('')
                                ax.set_xticklabels([])

                                # Y-label only on leftmost column
                                if pc_col_idx == 0:
                                    ax.set_ylabel(f'{layer.replace("_chunk60", "")}\n{ylabel}', fontsize=10)
                                else:
                                    ax.set_ylabel('')

                                # Apply per-feature y-axis limits if requested
                                if uniform_ylim and layer in feature_ylims:
                                    ax.set_ylim(*feature_ylims[layer])

                            # Overlay FDR asterisks + null-95 dashed lines on each bar
                            if has_sig:
                                _overlay_significance(bar_axes[layer_idx, pc_col_idx],
                                                      pc_data, layer, feature_order, value_col)
                                _overlay_significance(strip_axes[layer_idx, pc_col_idx],
                                                      pc_data, layer, feature_order, value_col)

                                ax.axhline(0, color='black', linewidth=1, alpha=0.5)

                                # Title only on top row
                                if layer_idx == 0:
                                    ax.set_title(f'PC {pc_index}', fontsize=12)

                    # Add overall title
                    bar_fig.suptitle(f'Pooled PCA Interpretation: {roi} ({roi_type}) - {network}',
                                    fontsize=16, fontweight='normal')
                    strip_fig.suptitle(f'Pooled PCA Interpretation: {roi} ({roi_type}) - {network}',
                                      fontsize=16, fontweight='normal')

                    # Create legend
                    feature_display = {'social': 'social interaction'}
                    legend_elements = [Patch(facecolor=colors_dict.get(feat, 'gray'),
                                             label=feature_display.get(feat, feat))
                                      for feat in feature_order if feat in network_data['label'].unique()]
                    if has_sig:
                        legend_elements.append(
                            Line2D([0], [0], color='black', linestyle='--',
                                   alpha=0.5, label='null 95%')
                        )

                    # Tight layout first (before adding legend)
                    plt.figure(bar_fig.number)
                    plt.tight_layout()
                    plt.figure(strip_fig.number)
                    plt.tight_layout()

                    # Add legend below with appropriate spacing
                    # Adjust bbox_to_anchor based on number of layers to prevent overlap
                    legend_y = -0.02 if n_layers > 1 else -0.15
                    bar_fig.legend(handles=legend_elements, loc='lower center',
                                  ncol=5, bbox_to_anchor=(0.5, legend_y), frameon=True, fontsize=10)
                    strip_fig.legend(handles=legend_elements, loc='lower center',
                                    ncol=5, bbox_to_anchor=(0.5, legend_y), frameon=True, fontsize=10)

                    # Save figures
                    Path(output_dir).mkdir(exist_ok=True, parents=True)

                    filename_base = f'pooled_pca_per_pc_{roi_type}_{roi}_{network}'

                    output_bar = os.path.join(output_dir, f'{filename_base}_bar.png')
                    bar_fig.savefig(output_bar, dpi=300, bbox_inches='tight')
                    print(f"    Saved: {output_bar}")

                    output_combined = os.path.join(output_dir, f'{filename_base}_combined.png')
                    strip_fig.savefig(output_combined, dpi=300, bbox_inches='tight')
                    print(f"    Saved: {output_combined}")

                    # Store figures
                    key = f"{roi_type}_{roi}_{network}"
                    figures[key] = {'bar': bar_fig, 'combined': strip_fig}

                    # Close to free memory
                    plt.close(bar_fig)
                    plt.close(strip_fig)

        print(f"\nPooled PCA interpretation plotting complete!")
        print(f"Output directory: {output_dir}")

        return figures

    def plot_pooled_pca_summary(
        self,
        results_csv_path=None,
        output_dir=None,
        fig_path=None,
        table_path=None,
        n_pcs=5,
        q_threshold=0.05,
        compress_xaxis=True,
        mask_nonsig=True,
        roi_type='func',
        image_model='alexnet',
        regions_to_plot=None,
        nonsig_style=None,
        ylabel=None,
        figsize=None,
        models_to_plot=None,
        orientation='v',
        hide_bars=False,
        show_region_label=True,
    ):
        """
        3×4 summary: max(r²) across PCs 0..n_pcs-1, restricted to FDR-significant PCs,
        per (region × model × feature). Non-significant → NaN (plotted gray).

        ``regions_to_plot`` keeps only those ROI keys, in the order given, and the
        figure loses a row per dropped region. ``nonsig_style`` overrides how
        non-significant bars are drawn -- 'gray' for solid grey (as in the
        compressed nonsig-shown figure) or 'dim' for the feature colour at low
        alpha (as in the all-features supplement) -- which otherwise follows from
        ``compress_xaxis``. ``ylabel`` overrides the y-axis label, whose default
        spells r-squared with a literal '²' that display faces often lack -- pass
        a mathtext version such as 'max $r^2$ of first 5 PCs' for those.
        ``figsize`` overrides the default of 21.5 x 5-per-row, which is sized for
        the compressed feature set -- widen it when plotting every feature.
        ``models_to_plot`` keeps only those model columns, in the order given.
        ``orientation='h'`` turns the panels on their side: features run down the
        y axis and r² along the x, which reads better when a single panel is
        carrying every feature. ``hide_bars`` draws the panel with its bars and
        asterisks invisible but everything else — axes, ticks, labels and the
        permutation null-95 lines — in place, giving a build-up frame that
        registers exactly with the full figure. ``show_region_label=False`` drops the
        bold region name, for a slide whose own title already names the region.
        """
        if orientation not in ('v', 'h'):
            raise ValueError(f"orientation must be 'v' or 'h'; got {orientation!r}")
        if nonsig_style not in (None, 'gray', 'dim'):
            raise ValueError(f"nonsig_style must be None, 'gray' or 'dim'; got {nonsig_style!r}")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.transforms import ScaledTranslation

        if results_csv_path is None:
            results_csv_path = os.path.join(
                self.dir, 'analysis', 'PCInterpretation',
                'compiled_pooled_results.csv'
            )
        if output_dir is None:
            # Compressed variants (sig-only or nonsig-shown) go to main;
            # the all-features version is supplemental.
            output_dir = os.path.join(
                self.dir, 'figures',
                'main' if compress_xaxis else 'supplemental'
            )
        if compress_xaxis and mask_nonsig:
            suffix = ''
        elif compress_xaxis and not mask_nonsig:
            suffix = '_nonsig_shown'
        else:
            suffix = '_all_features'
        # 'func' is the default — keep historical filenames; anat gets a tag
        roi_tag = '' if roi_type == 'func' else f'_{roi_type}'
        if image_model not in IMAGE_MODEL_SPECS:
            raise ValueError(f"Unknown image_model={image_model!r}; "
                             f"choose from {list(IMAGE_MODEL_SPECS)}")
        image_model_fs = IMAGE_MODEL_SPECS[image_model]
        if fig_path is None:
            fig_path = os.path.join(
                output_dir, f'pooled_pca_summary{roi_tag}{suffix}_{image_model}'
            )
        if table_path is None:
            table_path = os.path.join(
                output_dir, f'pooled_pca_summary{roi_tag}{suffix}_{image_model}.csv'
            )

        df = pd.read_csv(results_csv_path)

        if 'q_value_fdr' not in df.columns:
            raise ValueError("q_value_fdr column missing — rerun compile step with permutation null")

        # Fixed orderings
        if roi_type == 'func':
            regions = [
                ('MT',                 'motion\nresponsive\nMT'),
                ('STS',                'social\ninteraction\nselective\nSTS'),
                ('temporal_language',  'language\nselective\ntemporal'),
            ]
        else:
            regions = [
                ('MT',                 'MT (anatomical)'),
                ('STS',                'STS (anatomical)'),
                ('temporal_language',  'Temporal language\n(anatomical)'),
            ]
        if regions_to_plot is not None:
            by_key = dict(regions)
            missing = [k for k in regions_to_plot if k not in by_key]
            if missing:
                raise ValueError(f"unknown region key(s) {missing}; "
                                 f"choose from {list(by_key)}")
            regions = [(k, by_key[k]) for k in regions_to_plot]

        # (display_name, feature_space_string_in_csv)
        models = [
            ('Motion Model',  'motion'),
            ('Image Model',   image_model_fs),
            ('Speech Model',  'hubert_chunk120_layer10+hubert_chunk120_layer11+hubert_chunk120_layer12'),
            ('Sentence Model','GPT2_chunk120_layer10+GPT2_chunk120_layer11+GPT2_chunk120_layer12'),
        ]
        if models_to_plot is not None:
            by_name = dict((display, fs) for display, fs in models)
            missing = [m for m in models_to_plot if m not in by_name]
            if missing:
                raise ValueError(f"unknown model(s) {missing}; "
                                 f"choose from {list(by_name)}")
            models = [(m, by_name[m]) for m in models_to_plot]

        features = [
            'motion', 'pixel', 'hue',                                   # low-level visual
            'face', 'num_agents', 'written_text',                       # social-visual
            'amplitude', 'pitch',                                       # acoustic
            'arousal', 'valence', 'social', 'turn_taking',
            'mentalization',                                            # affective / social-semantic
        ]
        feature_display = {
            'num_agents': 'number of agents',
            'written_text': 'written text',
            'turn_taking': 'turn taking',
            'social': 'social interaction',
        }
        category_by_feature = {
            **{f: 'low-level visual'        for f in ['motion', 'pixel', 'hue']},
            **{f: 'social-visual'           for f in ['face', 'num_agents', 'written_text']},
            **{f: 'acoustic'                for f in ['amplitude', 'pitch']},
            **{f: 'affective/social-sem.'   for f in ['arousal', 'valence', 'social',
                                                     'turn_taking', 'mentalization']},
        }
        category_palettes = {
            'low-level visual':      ['#08306b', '#2171b5', '#6baed6'],
            'social-visual':         ['#00441b', '#238b45', '#74c476'],
            'acoustic':              ['#67000d', '#cb181d'],
            'affective/social-sem.': ['#3f007d', '#6a51a3', '#9e9ac8',
                                      '#c994c7', '#df65b0', '#e7298a'],
        }
        # Assign one color per feature (stable, category-coded)
        feature_colors = {}
        cat_counts = {}
        for feat in features:
            cat = category_by_feature[feat]
            i = cat_counts.get(cat, 0)
            feature_colors[feat] = category_palettes[cat][i]
            cat_counts[cat] = i + 1
        nonsig_color = '#d9d9d9'

        # Restrict to requested ROI type + PCs
        df = df[df['roi_type'] == roi_type]
        df = df[df['pc_index'] < n_pcs]

        # Per-subject duplicate rows exist, but q/r²/null are constant per
        # (roi, feature_space, pc_index, label) — take first.
        has_null95 = 'null_r2_95' in df.columns
        agg_kwargs = dict(r2=('r2_mean', 'first'),
                          q=('q_value_fdr', 'first'))
        if has_null95:
            agg_kwargs['null95'] = ('null_r2_95', 'first')
        agg = (df.groupby(['roi', 'feature_space', 'pc_index', 'label'],
                          as_index=False)
                 .agg(**agg_kwargs))

        # Build the (region × model × feature) table
        records = []
        for roi_key, _ in regions:
            for model_display, fs in models:
                sub = agg[(agg['roi'] == roi_key) & (agg['feature_space'] == fs)]
                for feat in features:
                    feat_rows = sub[sub['label'] == feat].dropna(subset=['r2'])
                    sig_rows = feat_rows[feat_rows['q'] < q_threshold]

                    def _row_null95(row):
                        return float(row['null95']) if has_null95 and pd.notna(row.get('null95', np.nan)) else np.nan

                    if len(sig_rows) > 0:
                        best = sig_rows.loc[sig_rows['r2'].idxmax()]
                        q = best['q']
                        if q < 0.001:
                            lvl = '***'
                        elif q < 0.01:
                            lvl = '**'
                        else:
                            lvl = '*'
                        records.append(dict(
                            region=roi_key, model=model_display, feature=feat,
                            max_sig_r2=float(best['r2']),
                            best_pc=int(best['pc_index']),
                            best_pc_q=float(q),
                            best_pc_null95=_row_null95(best),
                            best_pc_sig_level=lvl))
                    elif mask_nonsig or len(feat_rows) == 0:
                        # No significant PC → gray bar in main figure
                        records.append(dict(
                            region=roi_key, model=model_display, feature=feat,
                            max_sig_r2=np.nan, best_pc=np.nan, best_pc_q=np.nan,
                            best_pc_null95=np.nan, best_pc_sig_level=''))
                    else:
                        # Supplement: show the actual max r² across all PCs, no asterisk
                        best = feat_rows.loc[feat_rows['r2'].idxmax()]
                        records.append(dict(
                            region=roi_key, model=model_display, feature=feat,
                            max_sig_r2=float(best['r2']),
                            best_pc=int(best['pc_index']),
                            best_pc_q=float(best['q']) if pd.notna(best['q']) else np.nan,
                            best_pc_null95=_row_null95(best),
                            best_pc_sig_level=''))
        summary = pd.DataFrame.from_records(records)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary.to_csv(table_path, index=False)
        print(f"Saved summary table: {table_path}")

        # Optionally compress x-axis: drop features that are never significant anywhere.
        # Key off the significance level rather than max_sig_r2, so this works even when
        # mask_nonsig=False (which populates max_sig_r2 with non-sig r² values).
        if compress_xaxis:
            sig_any = (summary.groupby('feature')['best_pc_sig_level']
                              .apply(lambda s: (s.astype(str) != '').any()))
            kept_features = [f for f in features if sig_any.get(f, False)]
            dropped = [f for f in features if f not in kept_features]
            if dropped:
                print(f"  Dropping never-significant features: {dropped}")
            features = kept_features
        else:
            print("  Keeping all features (supplement version)")

        # Publication sizing: when plotting_helpers.figure_fontsize is set, every label is
        # drawn at that size with thin lines and the pdf is saved at figure_width_mm with
        # editable text (see plotting_helpers.save_figure). Otherwise label sizes are
        # absolute, scaled by the shared talk knob (1.0 leaves the defaults alone).
        publication = plotting_helpers.figure_fontsize is not None
        _fs = plotting_helpers.font_scale

        def fsz(default):
            return plotting_helpers.figure_fontsize if publication else default * _fs

        bar_lw = 0.25 if publication else 0.5
        null_lw = 0.5 if publication else 0.8
        if publication:
            plotting_helpers._update_font_rc(plotting_helpers.figure_fontsize)

        # Per-column (per-model) y-limits from the summary (include null-95 so it isn't clipped)
        col_ylims = {}
        for model_display, _ in models:
            msub = summary[summary['model'] == model_display]
            combined = pd.concat([msub['max_sig_r2'], msub['best_pc_null95']]).dropna()
            y_max = float(combined.max()) if len(combined) else 0.01
            col_ylims[model_display] = (0, y_max * 1.25 if y_max > 0 else 0.01)

        # Figure
        fig, axes = plt.subplots(len(regions), len(models),
                                 figsize=figsize or (21.5, 5 * len(regions)), squeeze=False,
                                 sharex='col' if orientation == 'v' else False,
                                 sharey='col' if orientation == 'h' else False)
        x = np.arange(len(features))
        for r_idx, (roi_key, region_label) in enumerate(regions):
            for c_idx, (model_display, _) in enumerate(models):
                ax = axes[r_idx, c_idx]
                panel = summary[(summary['region'] == roi_key)
                                & (summary['model'] == model_display)].set_index('feature')
                heights = [0 if np.isnan(panel.loc[f, 'max_sig_r2'])
                           else panel.loc[f, 'max_sig_r2'] for f in features]
                # nonsig-shown variant: non-sig bars are solid gray.
                # Full supplement: non-sig bars keep feature color, dimmed via alpha.
                is_sig = [bool(panel.loc[f, 'best_pc_sig_level']) for f in features]
                if nonsig_style is not None:
                    nonsig_shown_variant = nonsig_style == 'gray'
                else:
                    nonsig_shown_variant = compress_xaxis and not mask_nonsig
                if nonsig_shown_variant:
                    colors = [feature_colors[f] if sig else nonsig_color
                              for f, sig in zip(features, is_sig)]
                    alphas = [1.0 for _ in is_sig]
                else:
                    colors = [
                        feature_colors[f] if (not np.isnan(panel.loc[f, 'max_sig_r2']) or not mask_nonsig)
                        else nonsig_color
                        for f in features
                    ]
                    alphas = [1.0 if sig else 0.4 for sig in is_sig]

                feature_labels = [feature_display.get(f, f) for f in features]
                value_label = ylabel or f'max r² of first {n_pcs} PCs'
                lim_hi = col_ylims[model_display][1]

                if orientation == 'v':
                    bars = ax.bar(x, heights, color=colors, edgecolor='black', linewidth=bar_lw)
                else:
                    bars = ax.barh(x, heights, color=colors, edgecolor='black', linewidth=bar_lw)
                for bar, a in zip(bars, alphas):
                    bar.set_alpha(a)
                if hide_bars:
                    # Hidden rather than skipped, so the axes are scaled and laid
                    # out exactly as in the version that shows them.
                    for bar in bars:
                        bar.set_visible(False)

                # Supplement: draw per-bar null-95 reference line (dashed)
                if not mask_nonsig and has_null95:
                    thickness = ((bars[0].get_width() if orientation == 'v'
                                  else bars[0].get_height()) if len(bars) else 0.8)
                    for xi, f in zip(x, features):
                        null95 = panel.loc[f, 'best_pc_null95']
                        if pd.isna(null95):
                            continue
                        span = (xi - thickness / 2, xi + thickness / 2)
                        if orientation == 'v':
                            ax.hlines(null95, *span, linestyles='--', colors='black',
                                      alpha=0.6, linewidth=null_lw)
                        else:
                            ax.vlines(null95, *span, linestyles='--', colors='black',
                                      alpha=0.6, linewidth=null_lw)

                # Asterisks just beyond the end of each significant bar
                for xi, f in zip(x, features):
                    lvl = panel.loc[f, 'best_pc_sig_level']
                    if not lvl or hide_bars:
                        continue
                    end = panel.loc[f, 'max_sig_r2'] + lim_hi * 0.015
                    if orientation == 'v':
                        ax.text(xi, end, lvl, ha='center', va='bottom', fontsize=fsz(26))
                    else:
                        ax.text(end, xi, lvl, ha='left', va='center', fontsize=fsz(26))

                ax.spines[['top', 'right']].set_visible(False)
                if r_idx == 0:
                    ax.set_title(model_display, fontsize=fsz(30), fontweight='normal')

                if orientation == 'v':
                    ax.set_ylim(*col_ylims[model_display])
                    ax.set_xticks(x)
                    ax.set_xticklabels(feature_labels if r_idx == len(regions) - 1 else [],
                                       rotation=60, ha='right', fontsize=fsz(26))
                    # Rotated ha='right' labels visually sit slightly left of the tick — nudge right
                    # (by ~0.4 of the label's font size).
                    if r_idx == len(regions) - 1:
                        offset = ScaledTranslation(fsz(26) * 10 / 26 / 72, 0, fig.dpi_scale_trans)
                        for lbl in ax.get_xticklabels():
                            lbl.set_transform(lbl.get_transform() + offset)
                    ax.tick_params(axis='y', labelsize=fsz(20))
                    if c_idx == 0:
                        ax.set_ylabel(value_label, fontsize=fsz(26))
                        # Region label, sitting further left than the y-axis label
                        if show_region_label and publication:
                            # anchored to the y-axis label so the gap is fixed in points
                            ax.annotate(region_label, xy=(0, 0.5), xycoords=ax.yaxis.label,
                                        xytext=(-fsz(26), 0), textcoords='offset points',
                                        fontsize=fsz(24), ha='right', va='center', ma='center')
                        elif show_region_label:
                            ax.text(-0.6, 0.5, region_label, transform=ax.transAxes,
                                    rotation=0, fontweight='normal', fontsize=fsz(24),
                                    ha='center', va='center', ma='center')
                else:
                    ax.set_xlim(*col_ylims[model_display])
                    ax.set_yticks(x)
                    ax.set_yticklabels(feature_labels if c_idx == 0 else [],
                                       fontsize=fsz(26))
                    ax.invert_yaxis()          # first feature at the top
                    ax.tick_params(axis='x', labelsize=fsz(20))
                    if r_idx == len(regions) - 1:
                        ax.set_xlabel(value_label, fontsize=fsz(26))
                    if c_idx == 0 and show_region_label:
                        # The y ticks already carry the feature names, so the region
                        # goes above the panel rather than out to its left. Offset in
                        # inches, not axes fractions, so a tall panel does not open a
                        # gap between it and the model title it sits above.
                        clearance = 0.75 * _fs if r_idx == 0 else 0.1 * _fs
                        ax.text(0.5, 1.0, region_label,
                                transform=(ax.transAxes
                                           + ScaledTranslation(0, clearance,
                                                               fig.dpi_scale_trans)),
                                rotation=0, fontweight='normal',
                                fontsize=fsz(24), ha='center', va='bottom', ma='center')

        fig.tight_layout()

        # Supplemental variants get lower-resolution PNGs to keep the paper file lean.
        png_dpi = 300 if compress_xaxis else 150
        if plotting_helpers.figure_width_mm is not None:
            plotting_helpers.save_figure(fig, fig_path + '.pdf')  # exactly figure_width_mm wide
        else:
            fig.savefig(fig_path + '.pdf', bbox_inches='tight')
        fig.savefig(fig_path + '.png', dpi=png_dpi, bbox_inches='tight')
        print(f"Saved figure: {fig_path}.pdf / .png (dpi={png_dpi})")

        return fig, summary

    def plot_pooled_pca_summary_per_pc(
        self,
        results_csv_path=None,
        output_dir=None,
        fig_path=None,
        table_path=None,
        n_pcs=5,
        q_threshold=0.05,
        roi_type='func',
        style='pc_color',
        image_model='alexnet',
        regions_to_plot=None,
        models_to_plot=None,
        orientation='v',
        nonsig_style=None,
        ylabel=None,
        figsize=None,
        pcs_as_columns=False,
        star_fontsize=6,
        pc_label_base=0,
    ):
        """
        Supplement: 3×4 region × model grid showing all PCs separately.
        Asterisks mark FDR-significant cells.

        style:
          'pc_color' — features on x, PCs as grouped bars colored by PC index (viridis).
          'swap'     — PCs on x, features as grouped bars colored by feature (matches main fig).
          'expand'   — 15-row × 4-col grid; rows = (region × PC), each cell looks like the
                       main figure (one feature-colored bar per feature).

        ``regions_to_plot`` / ``models_to_plot`` restrict the grid, and (for the
        'expand' style) ``orientation='h'`` puts features on the y axis,
        ``nonsig_style='gray'`` draws non-significant bars solid grey rather than
        dimmed, and ``ylabel`` / ``figsize`` override the value-axis label and
        canvas. ``pcs_as_columns`` lays the PCs out across the page instead of
        stacking them, which keeps a single-region figure slide-shaped, in which
        case ``star_fontsize`` wants raising from the 6pt that suits the dense
        15-row grid. ``pc_label_base`` is the number shown for the first
        component -- 0 matches ``pc_index``, 1 counts them from PC 1 for an
        audience. All default to the published behaviour.
        """
        if orientation not in ('v', 'h'):
            raise ValueError(f"orientation must be 'v' or 'h'; got {orientation!r}")
        if nonsig_style not in (None, 'gray', 'dim'):
            raise ValueError(f"nonsig_style must be None, 'gray' or 'dim'; got {nonsig_style!r}")
        import matplotlib.pyplot as plt
        from matplotlib.transforms import ScaledTranslation

        if results_csv_path is None:
            results_csv_path = os.path.join(
                self.dir, 'analysis', 'PCInterpretation',
                'compiled_pooled_results.csv'
            )
        if output_dir is None:
            output_dir = os.path.join(self.dir, 'figures', 'supplemental')
        roi_tag = '' if roi_type == 'func' else f'_{roi_type}'
        style_tag = '' if style == 'pc_color' else f'_{style}'
        if image_model not in IMAGE_MODEL_SPECS:
            raise ValueError(f"Unknown image_model={image_model!r}; "
                             f"choose from {list(IMAGE_MODEL_SPECS)}")
        image_model_fs = IMAGE_MODEL_SPECS[image_model]
        if fig_path is None:
            fig_path = os.path.join(
                output_dir,
                f'pooled_pca_summary_per_pc{style_tag}{roi_tag}_{image_model}'
            )
        if table_path is None:
            table_path = os.path.join(
                output_dir,
                f'pooled_pca_summary_per_pc{style_tag}{roi_tag}_{image_model}.csv'
            )

        df = pd.read_csv(results_csv_path)
        if 'q_value_fdr' not in df.columns:
            raise ValueError("q_value_fdr column missing — rerun compile step")

        if roi_type == 'func':
            regions = [
                ('MT',                 'motion\nresponsive\nMT'),
                ('STS',                'social\ninteraction\nselective\nSTS'),
                ('temporal_language',  'language\nselective\ntemporal'),
            ]
        else:
            regions = [
                ('MT',                 'MT (anatomical)'),
                ('STS',                'STS (anatomical)'),
                ('temporal_language',  'Temporal language\n(anatomical)'),
            ]
        models = [
            ('Motion Model',  'motion'),
            ('Image Model',   image_model_fs),
            ('Speech Model',  'hubert_chunk120_layer10+hubert_chunk120_layer11+hubert_chunk120_layer12'),
            ('Sentence Model','GPT2_chunk120_layer10+GPT2_chunk120_layer11+GPT2_chunk120_layer12'),
        ]
        if regions_to_plot is not None:
            by_key = dict(regions)
            missing = [k for k in regions_to_plot if k not in by_key]
            if missing:
                raise ValueError(f"unknown region key(s) {missing}; choose from {list(by_key)}")
            regions = [(k, by_key[k]) for k in regions_to_plot]
        if models_to_plot is not None:
            by_name = dict(models)
            missing = [m for m in models_to_plot if m not in by_name]
            if missing:
                raise ValueError(f"unknown model(s) {missing}; choose from {list(by_name)}")
            models = [(m, by_name[m]) for m in models_to_plot]

        features = [
            'motion', 'pixel', 'hue',
            'face', 'num_agents', 'written_text',
            'amplitude', 'pitch',
            'arousal', 'valence', 'social', 'turn_taking',
            'mentalization',
        ]
        feature_display = {
            'num_agents': 'number of agents',
            'written_text': 'written text',
            'turn_taking': 'turn taking',
            'social': 'social interaction',
        }

        # PC color palette (sequential viridis-like)
        pc_cmap = plt.get_cmap('viridis')
        pc_colors = [pc_cmap(i / max(1, n_pcs - 1)) for i in range(n_pcs)]

        # Feature-color setup (matches main summary figure)
        category_by_feature = {
            **{f: 'low-level visual'      for f in ['motion', 'pixel', 'hue']},
            **{f: 'social-visual'         for f in ['face', 'num_agents', 'written_text']},
            **{f: 'acoustic'              for f in ['amplitude', 'pitch']},
            **{f: 'affective/social-sem.' for f in ['arousal', 'valence', 'social',
                                                   'turn_taking', 'mentalization']},
        }
        category_palettes = {
            'low-level visual':      ['#08306b', '#2171b5', '#6baed6'],
            'social-visual':         ['#00441b', '#238b45', '#74c476'],
            'acoustic':              ['#67000d', '#cb181d'],
            'affective/social-sem.': ['#3f007d', '#6a51a3', '#9e9ac8',
                                      '#c994c7', '#df65b0', '#e7298a'],
        }
        feature_colors = {}
        cat_counts = {}
        for feat in features:
            cat = category_by_feature[feat]
            i = cat_counts.get(cat, 0)
            feature_colors[feat] = category_palettes[cat][i]
            cat_counts[cat] = i + 1

        df = df[df['roi_type'] == roi_type]
        df = df[df['pc_index'] < n_pcs]
        has_null95 = 'null_r2_95' in df.columns
        agg_kwargs = dict(r2=('r2_mean', 'first'), q=('q_value_fdr', 'first'))
        if has_null95:
            agg_kwargs['null95'] = ('null_r2_95', 'first')
        agg = (df.groupby(['roi', 'feature_space', 'pc_index', 'label'], as_index=False)
                 .agg(**agg_kwargs))

        # Long-form summary table for CSV
        records = []
        for roi_key, _ in regions:
            for model_display, fs in models:
                sub = agg[(agg['roi'] == roi_key) & (agg['feature_space'] == fs)]
                for feat in features:
                    for pc in range(n_pcs):
                        row = sub[(sub['label'] == feat) & (sub['pc_index'] == pc)]
                        if len(row) == 0 or pd.isna(row.iloc[0]['r2']):
                            r2v, qv, lvl, null95 = np.nan, np.nan, '', np.nan
                        else:
                            r = row.iloc[0]
                            r2v, qv = float(r['r2']), float(r['q'])
                            null95 = float(r['null95']) if has_null95 and pd.notna(r.get('null95', np.nan)) else np.nan
                            if pd.notna(qv) and qv < q_threshold:
                                lvl = ('***' if qv < 0.001
                                       else '**' if qv < 0.01 else '*')
                            else:
                                lvl = ''
                        records.append(dict(region=roi_key, model=model_display,
                                            feature=feat, pc_index=pc,
                                            r2=r2v, q=qv, null95=null95,
                                            sig_level=lvl))
        summary = pd.DataFrame.from_records(records)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary.to_csv(table_path, index=False)
        print(f"Saved per-PC summary table: {table_path}")

        # Per-column ylim — max across PCs in that model column (incl. null-95 so it's never clipped)
        col_ylims = {}
        for model_display, _ in models:
            msub = summary[summary['model'] == model_display]
            cols = ['r2'] + (['null95'] if 'null95' in msub.columns else [])
            combined = pd.concat([msub[c] for c in cols]).dropna()
            y_max = float(combined.max()) if len(combined) else 0.01
            col_ylims[model_display] = (0, y_max * 1.20 if y_max > 0 else 0.01)

        n_feat = len(features)

        if style == 'pc_color':
            # Original: features on x, PCs as grouped bars colored by PC
            fig, axes = plt.subplots(len(regions), len(models),
                                     figsize=(16, 9), squeeze=False, sharex='col')
            x = np.arange(n_feat)
            bar_w = 0.8 / n_pcs
            for r_idx, (roi_key, region_label) in enumerate(regions):
                for c_idx, (model_display, _) in enumerate(models):
                    ax = axes[r_idx, c_idx]
                    panel = summary[(summary['region'] == roi_key)
                                    & (summary['model'] == model_display)]
                    for pc in range(n_pcs):
                        pc_panel = (panel[panel['pc_index'] == pc]
                                    .set_index('feature').reindex(features))
                        heights = pc_panel['r2'].fillna(0).values
                        offsets = (pc - (n_pcs - 1) / 2) * bar_w
                        bars = ax.bar(x + offsets, heights, width=bar_w,
                                      color=pc_colors[pc], edgecolor='black',
                                      linewidth=0.3)
                        for xi, feat, bar in zip(x, features, bars):
                            lvl = pc_panel.loc[feat, 'sig_level'] if feat in pc_panel.index else ''
                            if lvl:
                                ax.text(bar.get_x() + bar.get_width() / 2,
                                        bar.get_height() + col_ylims[model_display][1] * 0.01,
                                        lvl, ha='center', va='bottom', fontsize=6)
                    ax.set_ylim(*col_ylims[model_display])
                    ax.set_xticks(x)
                    x_labels = [feature_display.get(f, f) for f in features]
                    ax.set_xticklabels(x_labels if r_idx == len(regions) - 1 else [],
                                       rotation=60, ha='right', fontsize=18)
                    ax.tick_params(axis='y', labelsize=15)
                    ax.spines[['top', 'right']].set_visible(False)
                    if r_idx == 0:
                        ax.set_title(model_display, fontsize=22, fontweight='normal')
                    if c_idx == 0:
                        ax.set_ylabel('r²', fontsize=18)
                        ax.text(-0.45, 0.5, region_label, transform=ax.transAxes,
                                rotation=0, fontweight='normal', fontsize=16,
                                ha='center', va='center', ma='center')

        elif style == 'swap':
            # Option 1: PCs on x, features as grouped bars colored by feature (matches main fig)
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            # Publication sizing: when plotting_helpers.figure_fontsize is set, all text is
            # drawn at that size with thin lines (pdf saved at figure_width_mm, see below).
            publication = plotting_helpers.figure_fontsize is not None

            def fsz(default):
                return plotting_helpers.figure_fontsize if publication else default

            if publication:
                plotting_helpers._update_font_rc(plotting_helpers.figure_fontsize)
            fig, axes = plt.subplots(len(regions), len(models),
                                     figsize=figsize or (16, 9), squeeze=False, sharex='col')
            x = np.arange(n_pcs)
            # Fraction of the 1.0 spacing between PC positions consumed by the bar group.
            group_frac = 0.95
            bar_w = group_frac / n_feat
            for r_idx, (roi_key, region_label) in enumerate(regions):
                for c_idx, (model_display, _) in enumerate(models):
                    ax = axes[r_idx, c_idx]
                    panel = summary[(summary['region'] == roi_key)
                                    & (summary['model'] == model_display)]
                    for f_idx, feat in enumerate(features):
                        feat_panel = (panel[panel['feature'] == feat]
                                      .set_index('pc_index').reindex(range(n_pcs)))
                        heights = feat_panel['r2'].fillna(0).values
                        offsets = (f_idx - (n_feat - 1) / 2) * bar_w
                        bars = ax.bar(x + offsets, heights, width=bar_w,
                                      color=feature_colors[feat], edgecolor='black',
                                      linewidth=0.1 if publication else 0.2)
                        for xi, pc, bar in zip(x, range(n_pcs), bars):
                            lvl = (feat_panel.loc[pc, 'sig_level']
                                   if pc in feat_panel.index else '')
                            bar.set_alpha(1.0 if lvl else 0.4)
                            if has_null95:
                                null95 = feat_panel.loc[pc, 'null95'] if pc in feat_panel.index else np.nan
                                if pd.notna(null95):
                                    ax.hlines(null95,
                                              bar.get_x(), bar.get_x() + bar.get_width(),
                                              linestyles='--', colors='black',
                                              alpha=0.6, linewidth=0.3 if publication else 0.6)
                            if lvl:
                                ax.text(bar.get_x() + bar.get_width() / 2,
                                        bar.get_height() + col_ylims[model_display][1] * 0.01,
                                        lvl, ha='center', va='bottom', fontsize=fsz(5))
                    ax.set_ylim(*col_ylims[model_display])
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'PC {pc + pc_label_base}' for pc in range(n_pcs)]
                                       if r_idx == len(regions) - 1 else [], fontsize=fsz(14))
                    ax.tick_params(axis='y', labelsize=fsz(11))
                    ax.spines[['top', 'right']].set_visible(False)
                    if r_idx == 0:
                        ax.set_title(model_display, fontsize=fsz(17), fontweight='normal')
                    if c_idx == 0:
                        ax.set_ylabel(ylabel or 'r²', fontsize=fsz(14))
                        if publication:
                            # anchored to the y-axis label so the gap is fixed in points
                            ax.annotate(region_label, xy=(0, 0.5), xycoords=ax.yaxis.label,
                                        xytext=(-fsz(14), 0), textcoords='offset points',
                                        fontsize=fsz(13), ha='right', va='center', ma='center')
                        else:
                            ax.text(-0.45, 0.5, region_label, transform=ax.transAxes,
                                    rotation=0, fontweight='normal', fontsize=13,
                                    ha='center', va='center', ma='center')

            legend_elements = [Patch(facecolor=feature_colors[f],
                                     label=feature_display.get(f, f))
                               for f in features]
            if has_null95:
                legend_elements.append(
                    Line2D([0], [0], color='black', linestyle='--',
                           alpha=0.6, label='null 95%')
                )
            fig.legend(handles=legend_elements, loc='upper center',
                       ncol=min(len(legend_elements), 8),
                       bbox_to_anchor=(0.5, -0.01), frameon=True, fontsize=fsz(11))

        elif style == 'expand':
            # Option 2: rows = (region × PC), cols = models. Each cell mirrors the main fig.
            # With pcs_as_columns, the PCs move onto the x of the grid instead:
            # rows = regions, cols = (model × PC).
            _fs = plotting_helpers.font_scale
            nonsig_color = '#d9d9d9'
            n_rows = len(regions) if pcs_as_columns else len(regions) * n_pcs
            n_cols = len(models) * n_pcs if pcs_as_columns else len(models)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=figsize or (12, 1.6 * n_rows), squeeze=False,
                                     sharex='col' if orientation == 'v' else False,
                                     sharey='col' if orientation == 'h' else False)
            x = np.arange(n_feat)
            feature_labels = [feature_display.get(f, f) for f in features]
            if pcs_as_columns and len(models) == 1:
                fig.suptitle(models[0][0], fontsize=22*_fs, fontweight='normal')
            for r_idx, (roi_key, region_label) in enumerate(regions):
                for pc in range(n_pcs):
                    for c_idx, (model_display, _) in enumerate(models):
                        if pcs_as_columns:
                            row, col = r_idx, c_idx * n_pcs + pc
                        else:
                            row, col = r_idx * n_pcs + pc, c_idx
                        ax = axes[row, col]
                        cell = (summary[(summary['region'] == roi_key)
                                        & (summary['model'] == model_display)
                                        & (summary['pc_index'] == pc)]
                                .set_index('feature').reindex(features))
                        heights = cell['r2'].fillna(0).values
                        levels = [cell.loc[f, 'sig_level'] if f in cell.index else ''
                                  for f in features]
                        if nonsig_style == 'gray':
                            colors = [feature_colors[f] if lvl else nonsig_color
                                      for f, lvl in zip(features, levels)]
                            alphas = [1.0 for _ in levels]
                        else:
                            colors = [feature_colors[f] for f in features]
                            alphas = [1.0 if lvl else 0.4 for lvl in levels]
                        if orientation == 'v':
                            bars = ax.bar(x, heights, color=colors,
                                          edgecolor='black', linewidth=0.4)
                        else:
                            bars = ax.barh(x, heights, color=colors,
                                           edgecolor='black', linewidth=0.4)
                        lim_hi = col_ylims[model_display][1]
                        for xi, feat, bar, lvl, a in zip(x, features, bars, levels, alphas):
                            bar.set_alpha(a)
                            if has_null95:
                                null95 = cell.loc[feat, 'null95'] if feat in cell.index else np.nan
                                if pd.notna(null95):
                                    if orientation == 'v':
                                        ax.hlines(null95, bar.get_x(),
                                                  bar.get_x() + bar.get_width(),
                                                  linestyles='--', colors='black',
                                                  alpha=0.6, linewidth=0.7)
                                    else:
                                        ax.vlines(null95, bar.get_y(),
                                                  bar.get_y() + bar.get_height(),
                                                  linestyles='--', colors='black',
                                                  alpha=0.6, linewidth=0.7)
                            if not lvl:
                                continue
                            value = bar.get_height() if orientation == 'v' else bar.get_width()
                            if orientation == 'v':
                                ax.text(xi, value + lim_hi * 0.01, lvl,
                                        ha='center', va='bottom',
                                        fontsize=star_fontsize*_fs)
                            else:
                                ax.text(value + lim_hi * 0.01, xi, lvl,
                                        ha='left', va='center',
                                        fontsize=star_fontsize*_fs)
                        ax.spines[['top', 'right']].set_visible(False)
                        if row == 0:
                            if pcs_as_columns:
                                # One model gets a suptitle; several need naming per block.
                                title = (f'PC {pc + pc_label_base}' if len(models) == 1
                                         else f'{model_display}\nPC {pc + pc_label_base}')
                            else:
                                title = model_display
                            ax.set_title(title, fontsize=22*_fs, fontweight='normal')

                        if orientation == 'v':
                            ax.set_ylim(*col_ylims[model_display])
                            ax.set_xticks(x)
                            ax.set_xticklabels(feature_labels if row == n_rows - 1 else [],
                                               rotation=60, ha='right', fontsize=18*_fs)
                            ax.tick_params(axis='y', labelsize=15*_fs)
                            if col == 0:
                                ax.set_ylabel(ylabel or f'PC {pc + pc_label_base}\nr²',
                                              fontsize=17*_fs)
                                ax.text(-0.45, 0.5, region_label, transform=ax.transAxes,
                                        rotation=0, fontweight='normal', fontsize=15*_fs,
                                        ha='center', va='center', ma='center')
                        else:
                            ax.set_xlim(*col_ylims[model_display])
                            ax.set_yticks(x)
                            ax.set_yticklabels(feature_labels if col == 0 else [],
                                               fontsize=18*_fs)
                            ax.invert_yaxis()
                            ax.tick_params(axis='x', labelsize=15*_fs)
                            if row == n_rows - 1:
                                ax.set_xlabel(ylabel or 'r²', fontsize=17*_fs)
                            if col == 0:
                                # Feature names already occupy the left margin, so
                                # the region label goes in the axis-label slot outside
                                # them, and the PC identifies the panel from the top.
                                ax.set_ylabel(region_label, fontsize=17*_fs,
                                              fontweight='normal')
                                if not pcs_as_columns:
                                    ax.set_ylabel(f'PC {pc + pc_label_base}', fontsize=17*_fs,
                                                  fontweight='normal')
                                    if pc == 0:
                                        ax.text(0.5, 1.0, region_label,
                                                transform=(ax.transAxes
                                                           + ScaledTranslation(0, 0.75 * _fs,
                                                                               fig.dpi_scale_trans)),
                                                rotation=0, fontweight='normal', fontsize=15*_fs,
                                                ha='center', va='bottom', ma='center')

        else:
            raise ValueError(f"Unknown style={style!r}; choose 'pc_color', 'swap', or 'expand'")

        fig.tight_layout()
        # per-PC figures are always supplemental — use lower PNG DPI.
        if plotting_helpers.figure_width_mm is not None:
            plotting_helpers.save_figure(fig, fig_path + '.pdf')  # exactly figure_width_mm wide
        else:
            fig.savefig(fig_path + '.pdf', bbox_inches='tight')
        fig.savefig(fig_path + '.png', dpi=150, bbox_inches='tight')
        print(f"Saved figure: {fig_path}.pdf / .png (dpi=150)")

        return fig, summary


def main():
    parser = argparse.ArgumentParser(
        description='Huth 2016 style PCA on encoding model weights'
    )
    parser.add_argument('--dir', default=helpers.PROJECT_DIR)
    parser.add_argument('--out_dir', default='analysis')
    parser.add_argument('--model', default='vislang')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='List of subjects (e.g., sub-05 sub-06) or None for all')
    parser.add_argument('--features', nargs='+', default=None,
                       help='List of feature spaces or None for all')
    parser.add_argument('--rois', nargs='+',
                       default=['MT', 'STS', 'temporal_language', 'frontal'])
    parser.add_argument('--roi_types', nargs='+', default=['func'],
                       help="ROI types: 'func' (functional GLM), 'anat' (anatomical parcel), or both")
    parser.add_argument('--n_components', default=0.95, type=float,
                       help='Variance threshold (0-1) or number of components')
    parser.add_argument('--smoothing_fwhm', default=3.0, type=float)
    parser.add_argument('--chunklen', default=20, type=int)
    parser.add_argument('--mask', default='ISC')
    parser.add_argument('--space', default='MNI152NLin2009cAsym')
    parser.add_argument('--population', default='NT')
    parser.add_argument('--plot_pc_interpretation', action='store_true',
                       help='Plot PC interpretation results from analysis/PCInterpretation/')
    parser.add_argument('--plot_per_pc_interpretation', action='store_true',
                       help='Plot per-PC interpretation results from analysis/PCInterpretation/')
    parser.add_argument('--plot_pooled_pca', action='store_true',
                       help='Plot pooled PCA interpretation results from analysis/PCInterpretation/')
    parser.add_argument('--plot_summary', action='store_true',
                       help='Plot 3×4 region × model summary figure of max-significant r² across PCs')
    parser.add_argument('--plot_summary_supplement', action='store_true',
                       help='Supplement version: same summary figure with ALL features retained on x-axis')
    parser.add_argument('--plot_summary_anat', action='store_true',
                       help='3×4 summary figure restricted to anatomical ROIs')
    parser.add_argument('--plot_summary_supplement_anat', action='store_true',
                       help='Supplement (all features) version of the anatomical summary figure')
    parser.add_argument('--plot_summary_nonsig_shown', action='store_true',
                       help='Compressed x-axis but non-sig features shown as gray bars (func)')
    parser.add_argument('--plot_summary_nonsig_shown_anat', action='store_true',
                       help='Compressed x-axis but non-sig features shown as gray bars (anat)')
    parser.add_argument('--image_model', default='alexnet',
                       choices=list(IMAGE_MODEL_SPECS),
                       help='Which vision feature space populates the Image Model column '
                            'for pooled PCA summary figures')
    parser.add_argument('--plot_summary_per_pc', action='store_true',
                       help='Supplement: 3×4 grid showing all PCs separately as grouped bars (func ROIs)')
    parser.add_argument('--plot_summary_per_pc_anat', action='store_true',
                       help='Per-PC supplement figure restricted to anatomical ROIs')
    parser.add_argument('--plot_summary_per_pc_swap', action='store_true',
                       help='Per-PC supplement: PCs on x-axis, features as feature-colored bars (func)')
    parser.add_argument('--plot_summary_per_pc_swap_anat', action='store_true',
                       help='Per-PC supplement (anatomical, swap layout)')
    parser.add_argument('--plot_summary_per_pc_expand', action='store_true',
                       help='Per-PC supplement: 15-row × 4-col grid, each cell like the main figure (func)')
    parser.add_argument('--plot_summary_per_pc_expand_anat', action='store_true',
                       help='Per-PC supplement (anatomical, expand layout)')

    args = parser.parse_args()

    # Resolve paths
    args.out_dir = os.path.join(args.dir, args.out_dir)

    # Initialize analysis
    analyzer = WeightPCAAnalysis(args)

    # Check if we should plot PC interpretation results
    if args.plot_pc_interpretation:
        print("\n" + "="*60)
        print("PLOTTING PC INTERPRETATION RESULTS")
        print("="*60)
        analyzer.plot_pc_interpretation_results()
        return

    # Check if we should plot per-PC interpretation results
    if args.plot_per_pc_interpretation:
        print("\n" + "="*60)
        print("PLOTTING PER-PC INTERPRETATION RESULTS")
        print("="*60)
        analyzer.plot_pc_interpretation_per_pc_results()
        return

    # Check if we should plot pooled PCA results
    if args.plot_pooled_pca:
        print("\n" + "="*60)
        print("PLOTTING POOLED PCA INTERPRETATION RESULTS")
        print("="*60)
        analyzer.plot_pooled_pca_results()
        return

    if args.plot_summary:
        print("\n" + "="*60)
        print(f"PLOTTING 3×4 REGION × MODEL SUMMARY FIGURE [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary(image_model=args.image_model)
        return

    if args.plot_summary_supplement:
        print("\n" + "="*60)
        print(f"PLOTTING 3×4 SUMMARY FIGURE (SUPPLEMENT — ALL FEATURES) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary(
            compress_xaxis=False, mask_nonsig=False, image_model=args.image_model
        )
        return

    if args.plot_summary_anat:
        print("\n" + "="*60)
        print(f"PLOTTING 3×4 SUMMARY FIGURE (ANATOMICAL ROIs) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary(roi_type='anat', image_model=args.image_model)
        return

    if args.plot_summary_supplement_anat:
        print("\n" + "="*60)
        print(f"PLOTTING 3×4 SUMMARY FIGURE (ANATOMICAL ROIs — ALL FEATURES) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary(
            roi_type='anat', compress_xaxis=False, mask_nonsig=False,
            image_model=args.image_model,
        )
        return

    if args.plot_summary_nonsig_shown:
        print("\n" + "="*60)
        print(f"PLOTTING 3×4 SUMMARY FIGURE (NON-SIG BARS SHOWN, GRAY) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary(
            compress_xaxis=True, mask_nonsig=False, image_model=args.image_model
        )
        return

    if args.plot_summary_nonsig_shown_anat:
        print("\n" + "="*60)
        print(f"PLOTTING 3×4 SUMMARY FIGURE (ANATOMICAL — NON-SIG BARS SHOWN, GRAY) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary(
            roi_type='anat', compress_xaxis=True, mask_nonsig=False,
            image_model=args.image_model,
        )
        return

    if args.plot_summary_per_pc:
        print("\n" + "="*60)
        print(f"PLOTTING 3×4 PER-PC SUMMARY FIGURE [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary_per_pc(image_model=args.image_model)
        return

    if args.plot_summary_per_pc_anat:
        print("\n" + "="*60)
        print(f"PLOTTING 3×4 PER-PC SUMMARY FIGURE (ANATOMICAL ROIs) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary_per_pc(
            roi_type='anat', image_model=args.image_model
        )
        return

    if args.plot_summary_per_pc_swap:
        print("\n" + "="*60)
        print(f"PLOTTING PER-PC SUMMARY (SWAP LAYOUT, FEATURE COLORS) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary_per_pc(
            style='swap', image_model=args.image_model
        )
        return

    if args.plot_summary_per_pc_swap_anat:
        print("\n" + "="*60)
        print(f"PLOTTING PER-PC SUMMARY (SWAP LAYOUT, ANATOMICAL) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary_per_pc(
            style='swap', roi_type='anat', image_model=args.image_model
        )
        return

    if args.plot_summary_per_pc_expand:
        print("\n" + "="*60)
        print(f"PLOTTING PER-PC SUMMARY (EXPANDED 15×4 GRID, FEATURE COLORS) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary_per_pc(
            style='expand', image_model=args.image_model
        )
        return

    if args.plot_summary_per_pc_expand_anat:
        print("\n" + "="*60)
        print(f"PLOTTING PER-PC SUMMARY (EXPANDED 15×4 GRID, ANATOMICAL) [image_model={args.image_model}]")
        print("="*60)
        analyzer.plot_pooled_pca_summary_per_pc(
            style='expand', roi_type='anat', image_model=args.image_model
        )
        return

    # Run batch analysis
    summary_df = analyzer.analyze_all_feature_spaces(
        subjects=args.subjects,
        feature_spaces=args.features,
        rois=args.rois,
        roi_types=args.roi_types,
        n_components=args.n_components
    )

    print("\nAnalysis complete!")

    # Print summary statistics for successful analyses
    successful = summary_df[summary_df['status'] == 'success']
    if len(successful) > 0 and 'n_components' in successful.columns:
        print("\nMean components by ROI and feature:")
        print(successful.groupby(['roi', 'feature_space'])['n_components'].mean())
        print("\nMean variance explained:")
        print(successful.groupby(['roi', 'feature_space'])['variance_explained'].mean())
    else:
        print("No successful analyses to summarize.")


if __name__ == '__main__':
    main()
