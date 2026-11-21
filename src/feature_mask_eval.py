# I/O packages
import os
from pathlib import Path
from argparse import Namespace

# data manipulation packages
import numpy as np
import pandas as pd
import nibabel

# plotting packages
import matplotlib.pyplot as plt
import matplotlib

# custom python modules
from src import helpers, plotting_helpers

CONDITIONS = ['high', 'low', 'contrast']

MODEL_FEATURE_LABELS = {
    'alexnet_eps0.3': 'AlexNet',
    'alexnet': 'AlexNet',
    'motion': 'Motion',
    'hubert_chunk120': 'HuBERT',
    'hubert_chunk60': 'HuBERT',
    'hubert': 'HuBERT',
    'word2vec_chunk120': 'Word2Vec',
    'word2vec_chunk60': 'Word2Vec',
    'word2vec': 'Word2Vec',
    'GPT2_chunk120': 'GPT-2',
    'GPT2_chunk60': 'GPT-2',
    'GPT2': 'GPT-2',
    'sbert': 'sBERT',
    'SimCLR': 'SimCLR',
    'pixel': 'Pixel',
    'hue': 'Hue',
}

BRAIN_SHAPE = (97, 115, 97)


def _build_file_label(model, smoothing_fwhm, mask, chunklen):
    """Build the file label string matching SecondLevelGroup's file_label_new pattern."""
    if model == 'cross_subject':
        file_label = f'_brain2brain_encoding_smoothingfwhm-{smoothing_fwhm}'
    elif model == 'correlation':
        file_label = f'_brain2brain_correlation_smoothingfwhm-{smoothing_fwhm}'
    else:
        file_label = f'_encoding_model-{model}_smoothingfwhm-{smoothing_fwhm}'

    if mask != '':
        file_label = file_label + f'_mask-{mask}_chunklen-{chunklen}'
    else:
        file_label = file_label + f'_chunklen-{chunklen}'

    return file_label


def _build_file_label_old(model, smoothing_fwhm, mask, chunklen):
    """Build old-style file label (chunklen before mask) for backward compatibility."""
    if model == 'cross_subject':
        file_label = f'_brain2brain_encoding_smoothingfwhm-{smoothing_fwhm}_chunklen-{chunklen}'
    elif model == 'correlation':
        file_label = f'_brain2brain_correlation_smoothingfwhm-{smoothing_fwhm}_chunklen-{chunklen}'
    else:
        file_label = f'_encoding_model-{model}_smoothingfwhm-{smoothing_fwhm}_chunklen-{chunklen}'

    if mask != '':
        file_label = file_label + f'_mask-{mask}'

    return file_label


def _load_flexible(base_dir, subfolder, subject, file_label_new, file_label_old, measure_label):
    """Try loading a file with new then old file label ordering."""
    for fl in [file_label_new, file_label_old]:
        filepath = os.path.join(base_dir, subfolder, f"{subject}{fl}_measure-{measure_label}.nii.gz")
        if os.path.exists(filepath):
            return nibabel.load(filepath)

    raise FileNotFoundError(
        f"Could not find file for {subject} with measure {measure_label} in {base_dir}/{subfolder}"
    )


def _load_flexible_csv(base_dir, subject, file_label_new, file_label_old, csv_suffix):
    """Try loading a CSV with new then old file label ordering."""
    for fl in [file_label_new, file_label_old]:
        filepath = os.path.join(base_dir, f"{subject}{fl}{csv_suffix}")
        if os.path.exists(filepath):
            return pd.read_csv(filepath), filepath

    raise FileNotFoundError(
        f"Could not find CSV for {subject} with suffix {csv_suffix} in {base_dir}"
    )


def load_subset_eval(analysis_dir, sid, file_label, measure, condition):
    """Load a subset eval NIfTI file.

    Parameters
    ----------
    analysis_dir : str
        Path to SecondLevelGroup output directory.
    sid : str
        Subject/group id, e.g. 'sub-NT'.
    file_label : str
        File label string (from _build_file_label).
    measure : str
        'perf' or 'ind_product_measure'.
    condition : str
        'high', 'low', or 'contrast'.

    Returns
    -------
    data : np.ndarray
        perf: (n_eval_features, x, y, z)
        ind_product_measure: (n_eval_features, n_model_features, x, y, z)
    affine : np.ndarray
    """
    if measure == 'perf':
        subfolder = 'performance'
        filename = f'{sid}{file_label}_measure-perf_subset_{condition}_eval_raw.nii.gz'
    else:
        subfolder = 'ind_product_measure'
        filename = f'{sid}{file_label}_measure-ind_product_measure_subset_{condition}_eval_raw.nii.gz'

    filepath = os.path.join(analysis_dir, subfolder, filename)
    nii = nibabel.load(filepath)
    return nii.get_fdata(), nii.affine


def compile_subset_eval_data(base_dir, out_dir, model, smoothing_fwhm, mask, chunklen,
                             population='NT', task='sherlock', conditions=None, load=False):
    """Compile per-subject subset eval files into group-level averages.

    Parameters
    ----------
    base_dir : str
        Top-level project directory.
    out_dir : str
        Analysis output directory (parent of EncodingModel/, SecondLevelGroup/).
    model : str
        Encoding model name, e.g. 'joint'.
    smoothing_fwhm : float
        Smoothing FWHM.
    mask : str
        Mask name, e.g. 'ISC'.
    chunklen : int
        Chunk length.
    population : str
        'NT'.
    task : str
        Task name for subject lookup.
    load : bool
        If True, skip compilation and load existing group-level files.

    Returns
    -------
    features_df : pd.DataFrame
        The eval features metadata.
    """
    file_label_new = _build_file_label(model, smoothing_fwhm, mask, chunklen)
    sid_group = f'sub-{population}'
    group_dir = os.path.join(out_dir, 'SecondLevelGroup')
    in_dir = os.path.join(out_dir, 'EncodingModel')

    if load:
        features_df = _load_features_df(out_dir, model, sid_group, file_label_new)
        print('Loaded existing group subset eval data (load=True)')
        return features_df

    subjects_dict = helpers.get_subjects(population)
    subjects = subjects_dict[task]

    file_label_old = _build_file_label_old(model, smoothing_fwhm, mask, chunklen)

    Path(os.path.join(group_dir, 'performance')).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(group_dir, 'ind_product_measure')).mkdir(exist_ok=True, parents=True)

    # Load subset eval features CSV
    # New location: <out_dir>/EncodingModel/features/encoding_model-<model>_features.csv
    # Fallback: per-subject file with full file label
    features_df = None
    features_csv_path = os.path.join(in_dir, 'features', f'encoding_model-{model}_subset_eval_features.csv')
    if os.path.exists(features_csv_path):
        features_df = pd.read_csv(features_csv_path)
    else:
        # Fallback: try per-subject CSVs with old naming
        for subject in subjects:
            try:
                features_df, _ = _load_flexible_csv(
                    in_dir, subject, file_label_new, file_label_old, '_subset_eval_features.csv')
                break
            except FileNotFoundError:
                continue

    if features_df is None:
        raise FileNotFoundError(
            f"Could not find subset_eval_features.csv.\n"
            f"  Tried: {features_csv_path}\n"
            f"  Also tried per-subject files in {in_dir}"
        )

    # Save group-level copy of features CSV
    features_df.to_csv(os.path.join(group_dir, f'{sid_group}{file_label_new}_subset_eval_features.csv'),
                       index=False)

    # Compile performance and product measure for each condition
    # Use incremental averaging to avoid loading all subjects into memory at once
    if conditions is None:
        conditions = CONDITIONS
    for condition in conditions:
        # --- Performance ---
        perf_measure_label = f'perf_subset_{condition}_eval_raw'
        sum_perf = None
        n_perf = 0
        affine = None
        for subject in subjects:
            try:
                nii = _load_flexible(in_dir, 'performance', subject,
                                     file_label_new, file_label_old, perf_measure_label)
                data = nii.get_fdata()
                if sum_perf is None:
                    sum_perf = np.nan_to_num(data, nan=0.0)
                else:
                    sum_perf += np.nan_to_num(data, nan=0.0)
                n_perf += 1
                if affine is None:
                    affine = nii.affine
            except FileNotFoundError:
                print(f'Warning: missing perf_subset_{condition} for {subject}')

        if n_perf > 0:
            avg_perf = sum_perf / n_perf
            img = nibabel.Nifti1Image(avg_perf, affine)
            nibabel.save(img, os.path.join(
                group_dir, 'performance',
                f'{sid_group}{file_label_new}_measure-{perf_measure_label}.nii.gz'))
            print(f'compiled: perf_subset_{condition} from {n_perf} subjects, shape {avg_perf.shape}')
            del sum_perf, avg_perf

        # --- Product measure (masked per-subject by performance > 0) ---
        pm_measure_label = f'ind_product_measure_subset_{condition}_eval_raw'
        sum_pm = None
        n_pm = 0
        for subject in subjects:
            try:
                nii = _load_flexible(in_dir, 'ind_product_measure', subject,
                                     file_label_new, file_label_old, pm_measure_label)
                data = np.nan_to_num(nii.get_fdata(), nan=0.0)
                # Mask by this subject's own performance: zero out where perf < 0
                try:
                    perf_nii = _load_flexible(in_dir, 'performance', subject,
                                              file_label_new, file_label_old, perf_measure_label)
                    perf_data = perf_nii.get_fdata()
                    # perf_data: (n_eval_features, x, y, z)
                    # data:      (n_eval_features, n_model_features, x, y, z)
                    # For each eval feature, zero out voxels where perf < 0
                    for ei in range(data.shape[0]):
                        data[ei, :, perf_data[ei] < 0] = 0
                    del perf_nii, perf_data
                except FileNotFoundError:
                    print(f'Warning: missing perf for {subject}, skipping per-subject masking')
                if sum_pm is None:
                    sum_pm = data
                else:
                    sum_pm += data
                n_pm += 1
                if affine is None:
                    affine = nii.affine
            except FileNotFoundError:
                print(f'Warning: missing ind_product_measure_subset_{condition} for {subject}')

        if n_pm > 0:
            avg_pm = sum_pm / n_pm
            img = nibabel.Nifti1Image(avg_pm, affine)
            nibabel.save(img, os.path.join(
                group_dir, 'ind_product_measure',
                f'{sid_group}{file_label_new}_measure-{pm_measure_label}.nii.gz'))
            print(f'compiled: ind_product_measure_subset_{condition} from {n_pm} subjects, shape {avg_pm.shape}')
            del sum_pm, avg_pm

    return features_df


def plot_subset_eval_performance(analysis_dir, figure_dir, sid, file_label, features_df,
                                 conditions=None,
                                 threshold=0.00001, vmax_high_low=None, vmax_contrast=None,
                                 cmap_positive='yellow_hot', cmap_contrast='blue_neg_yellow_pos',
                                 ROI_niis=[], ROI_names=[], ROI_colors=[]):
    """Plot brain surface maps for subset eval performance (high, low, contrast).

    Parameters
    ----------
    analysis_dir : str
        Path to SecondLevelGroup output directory.
    figure_dir : str
        Path to figure output directory.
    sid : str
        Subject/group id, e.g. 'sub-NT'.
    file_label : str
        File label from _build_file_label.
    features_df : pd.DataFrame
        Eval features metadata with 'name' column.
    threshold : float
        Plotting threshold for surface maps.
    vmax_high_low : float or None
        Max value for high/low condition plots.
    vmax_contrast : float or None
        Max value for contrast plots.
    cmap_positive : str
        Colormap name for high/low (non-negative R-squared).
    cmap_contrast : str
        Colormap name for contrast (signed difference).
    ROI_niis : list
        List of nibabel NIfTI ROI images for overlay.
    ROI_names : list
        ROI names corresponding to ROI_niis.
    ROI_colors : list
        Colors corresponding to ROI_niis.
    """
    out_dir = os.path.join(figure_dir, 'performance')
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    feature_names = list(features_df['name'])

    if conditions is None:
        conditions = CONDITIONS
    for condition in conditions:
        data, affine = load_subset_eval(analysis_dir, sid, file_label, 'perf', condition)

        if condition == 'contrast':
            cmap = cmap_contrast
            symmetric_cbar = True
            vmax = vmax_contrast
            colorbar_label = r'$\Delta$ Explained Variance $R^2$ (high $-$ low)'
        else:
            cmap = cmap_positive
            symmetric_cbar = False
            vmax = vmax_high_low
            colorbar_label = 'Explained Variance $R^2$'

        for i, feat_name in enumerate(feature_names):
            vol = data[i]
            nii = nibabel.Nifti1Image(vol, affine)

            filepath = os.path.join(out_dir, f'{sid}{file_label}_feature-{feat_name}_perf_{condition}')
            title = f'{feat_name} ({condition})'

            plotting_helpers.plot_surface(
                nii, filepath,
                ROI_niis=ROI_niis,
                threshold=threshold,
                vmax=vmax,
                cmap=cmap,
                title=title,
                symmetric_cbar=symmetric_cbar,
                colorbar_label=colorbar_label,
                ROIs=ROI_names,
                ROI_colors=ROI_colors)
            print(f'plotted: perf {feat_name} {condition}')


def plot_subset_eval_product_measure(analysis_dir, figure_dir, sid, file_label, features_df,
                                     model_feature_names,
                                     conditions=None,
                                     threshold=0.00001, vmax_high_low=None, vmax_contrast=None,
                                     cmap_positive='yellow_hot', cmap_contrast='blue_neg_yellow_pos',
                                     ROI_niis=[], ROI_names=[], ROI_colors=[],
                                     mask_by_performance=True,
                                     group_layers=False, raw_model_feature_names=None):
    """Plot brain surface maps for subset eval product measure (high, low, contrast).

    Parameters
    ----------
    analysis_dir : str
        Path to SecondLevelGroup output directory.
    figure_dir : str
        Path to figure output directory.
    sid : str
        Subject/group id, e.g. 'sub-NT'.
    file_label : str
        File label from _build_file_label.
    features_df : pd.DataFrame
        Eval features metadata with 'name' column.
    model_feature_names : list of str
        Names of the model features (columns of the product measure).
    threshold : float
        Plotting threshold for surface maps.
    vmax_high_low : float or None
        Max value for high/low condition plots.
    vmax_contrast : float or None
        Max value for contrast plots.
    cmap_positive : str
        Colormap name for high/low (non-negative R-squared).
    cmap_contrast : str
        Colormap name for contrast (signed difference).
    ROI_niis : list
        List of nibabel NIfTI ROI images for overlay.
    ROI_names : list
        ROI names corresponding to ROI_niis.
    ROI_colors : list
        Colors corresponding to ROI_niis.
    mask_by_performance : bool
        If True, zero out voxels where overall performance < 0.
    """
    eval_feature_names = list(features_df['name'])

    # Build layer grouping if needed
    if group_layers and raw_model_feature_names is not None:
        _, group_indices = _get_layer_grouping(raw_model_feature_names)
    else:
        group_indices = None

    if conditions is None:
        conditions = CONDITIONS
    for condition in conditions:
        pm_data, affine = load_subset_eval(
            analysis_dir, sid, file_label, 'ind_product_measure', condition)

        # Aggregate layers before plotting
        if group_indices is not None:
            pm_data = _aggregate_product_measure_layers(pm_data, group_indices)

        # Optionally mask by performance: zero out voxels with negative perf
        if mask_by_performance:
            try:
                perf_data, _ = load_subset_eval(
                    analysis_dir, sid, file_label, 'perf', condition)
            except FileNotFoundError:
                perf_data = None

        if condition == 'contrast':
            cmap = cmap_contrast
            symmetric_cbar = True
            vmax = vmax_contrast
            colorbar_label = r'$\Delta$ Explained Variance $R^2$ (high $-$ low)'
        else:
            cmap = cmap_positive
            symmetric_cbar = False
            vmax = vmax_high_low
            colorbar_label = 'Explained Variance $R^2$'

        for i, eval_feat in enumerate(eval_feature_names):
            out_dir = os.path.join(figure_dir, 'ind_product_measure', eval_feat)
            Path(out_dir).mkdir(exist_ok=True, parents=True)

            for j, model_feat in enumerate(model_feature_names):
                vol = pm_data[i, j].copy()

                # Mask voxels with negative performance
                if mask_by_performance and perf_data is not None:
                    vol[perf_data[i] < 0] = 0

                nii = nibabel.Nifti1Image(vol, affine)

                filepath = os.path.join(
                    out_dir,
                    f'{sid}{file_label}_evalfeat-{eval_feat}_modelfeat-{model_feat}_{condition}')
                title = f'{eval_feat} / {model_feat} ({condition})'

                plotting_helpers.plot_surface(
                    nii, filepath,
                    ROI_niis=ROI_niis,
                    threshold=threshold,
                    vmax=vmax,
                    cmap=cmap,
                    title=title,
                    symmetric_cbar=symmetric_cbar,
                    colorbar_label=colorbar_label,
                    ROIs=ROI_names,
                    ROI_colors=ROI_colors)
                print(f'plotted: product_measure {eval_feat}/{model_feat} {condition}')

        # Free large arrays after each condition to prevent memory accumulation
        del pm_data
        if mask_by_performance:
            try:
                del perf_data
            except NameError:
                pass
        import gc; gc.collect()


def plot_subset_eval_roi_summary(analysis_dir, figure_dir, sid, file_label, features_df,
                                 roi_dict, measure='perf', model_feature_names=None,
                                 conditions=None, mask_by_performance=True):
    """Create bar plots summarizing high/low/contrast values within ROIs.

    Parameters
    ----------
    analysis_dir : str
        Path to SecondLevelGroup output directory.
    figure_dir : str
        Path to figure output directory.
    sid : str
        Subject/group id.
    file_label : str
        File label from _build_file_label.
    features_df : pd.DataFrame
        Eval features metadata with 'name' column.
    roi_dict : dict
        Mapping from ROI name (str) to nibabel Nifti1Image mask.
    measure : str
        'perf' or 'ind_product_measure'.
    model_feature_names : list of str or None
        Required if measure='ind_product_measure'.
    mask_by_performance : bool
        If True and measure='ind_product_measure', mask by positive performance.
    """
    out_dir = os.path.join(figure_dir, 'roi_summary')
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    eval_feature_names = list(features_df['name'])
    roi_names = list(roi_dict.keys())

    # Load data for all conditions
    if conditions is None:
        conditions = CONDITIONS
    data_by_condition = {}
    perf_by_condition = {}
    for condition in conditions:
        data_by_condition[condition], _ = load_subset_eval(
            analysis_dir, sid, file_label, measure, condition)
        if measure == 'ind_product_measure' and mask_by_performance:
            try:
                perf_by_condition[condition], _ = load_subset_eval(
                    analysis_dir, sid, file_label, 'perf', condition)
            except FileNotFoundError:
                perf_by_condition[condition] = None

    if measure == 'perf':
        _plot_roi_summary_perf(
            data_by_condition, eval_feature_names, roi_dict, roi_names, out_dir,
            sid, file_label)
    else:
        _plot_roi_summary_product_measure(
            data_by_condition, perf_by_condition, eval_feature_names, roi_dict,
            roi_names, model_feature_names, out_dir, sid, file_label,
            mask_by_performance)


def _plot_roi_summary_perf(data_by_condition, eval_feature_names, roi_dict, roi_names,
                           out_dir, sid, file_label):
    """Bar plot of mean performance within each ROI for each eval feature x condition."""
    n_rois = len(roi_names)
    n_features = len(eval_feature_names)

    fig, axes = plt.subplots(1, n_rois, figsize=(5 * n_rois, 5), squeeze=False)
    axes = axes[0]

    bar_width = 0.25
    condition_colors = {'high': '#e67e22', 'low': '#3498db', 'contrast': '#2ecc71'}
    x = np.arange(n_features)

    for roi_idx, roi_name in enumerate(roi_names):
        ax = axes[roi_idx]
        roi_mask = roi_dict[roi_name].get_fdata()
        roi_mask_bool = roi_mask > 0

        conditions = list(data_by_condition.keys())
        for c_idx, condition in enumerate(conditions):
            means = []
            for f_idx in range(n_features):
                vol = data_by_condition[condition][f_idx]
                roi_vals = vol[roi_mask_bool]
                means.append(np.nanmean(roi_vals))

            ax.bar(x + c_idx * bar_width, means, bar_width,
                   label=condition, color=condition_colors.get(condition, 'gray'), alpha=0.85)

        ax.set_title(roi_name, fontsize=14, fontweight='normal')
        ax.set_xticks(x + bar_width * (len(conditions) - 1) / 2)
        ax.set_xticklabels(eval_feature_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Mean $R^2$', fontsize=12)
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    filepath = os.path.join(out_dir, f'{sid}{file_label}_perf_roi_summary.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'saved: {filepath}')


def _plot_roi_summary_product_measure(data_by_condition, perf_by_condition,
                                      eval_feature_names, roi_dict, roi_names,
                                      model_feature_names, out_dir, sid, file_label,
                                      mask_by_performance):
    """Bar plots of mean product measure within ROIs, one figure per model feature."""
    n_rois = len(roi_names)
    n_features = len(eval_feature_names)

    bar_width = 0.25
    condition_colors = {'high': '#e67e22', 'low': '#3498db', 'contrast': '#2ecc71'}
    x = np.arange(n_features)

    for m_idx, model_feat in enumerate(model_feature_names):
        fig, axes = plt.subplots(1, n_rois, figsize=(5 * n_rois, 5), squeeze=False)
        axes = axes[0]

        for roi_idx, roi_name in enumerate(roi_names):
            ax = axes[roi_idx]
            roi_mask = roi_dict[roi_name].get_fdata()
            roi_mask_bool = roi_mask > 0

            conditions = list(data_by_condition.keys())
            for c_idx, condition in enumerate(conditions):
                means = []
                for f_idx in range(n_features):
                    vol = data_by_condition[condition][f_idx, m_idx].copy()

                    if mask_by_performance and perf_by_condition.get(condition) is not None:
                        perf_vol = perf_by_condition[condition][f_idx]
                        vol[perf_vol < 0] = 0

                    roi_vals = vol[roi_mask_bool]
                    means.append(np.nanmean(roi_vals))

                ax.bar(x + c_idx * bar_width, means, bar_width,
                       label=condition, color=condition_colors.get(condition, 'gray'), alpha=0.85)

            ax.set_title(f'{roi_name}\n{model_feat}', fontsize=12, fontweight='normal')
            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(eval_feature_names, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Mean product measure $R^2$', fontsize=12)
            ax.legend(fontsize=10)
            ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        filepath = os.path.join(
            out_dir, f'{sid}{file_label}_ind_product_measure_roi_summary_{model_feat}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'saved: {filepath}')


def _load_roi_masks(base_dir, roi_names):
    """Load ROI masks and return dict and list for plotting."""
    import nilearn.image
    loader = Namespace(dir=base_dir, brain_shape=BRAIN_SHAPE)
    colors_dict = plotting_helpers.get_colors_dict()
    default_roi_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    roi_dict = {}
    roi_niis = []
    roi_colors_list = []
    for i, roi_name in enumerate(roi_names):
        roi_nii = helpers.load_mask(loader, roi_name)
        roi_dict[roi_name] = roi_nii
        roi_niis.append(roi_nii)
        roi_colors_list.append(
            colors_dict.get(roi_name, default_roi_colors[i % len(default_roi_colors)]))
    return roi_dict, roi_niis, roi_colors_list


def _load_model_feature_names(out_dir, model, group_layers=False):
    """Load model feature names from the features CSV.

    Parameters
    ----------
    out_dir : str
        Analysis output directory.
    model : str
        Encoding model name.
    group_layers : bool
        If True, collapse layered features (e.g. 'alexnet_layer1', 'alexnet_layer2')
        into a single grouped name (e.g. 'alexnet').
    """
    import csv
    features_path = os.path.join(out_dir, 'EncodingModel', 'features', f'encoding_model-{model}_features.csv')
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            names = list(csv.reader(f, delimiter=','))[0]
    else:
        # Fallback: split model name on underscores
        names = model.split('_')

    if group_layers:
        names = helpers.group_network_layers(names)

    return names


def _get_layer_grouping(raw_feature_names):
    """Build a mapping from grouped feature names to indices in the raw feature list.

    Parameters
    ----------
    raw_feature_names : list of str
        Raw feature names, e.g. ['alexnet_layer1', 'alexnet_layer2', ..., 'motion'].

    Returns
    -------
    grouped_names : list of str
        Grouped feature names, e.g. ['alexnet', 'motion'].
    group_indices : list of list of int
        For each grouped name, the indices into raw_feature_names.
    """
    grouped_names = helpers.group_network_layers(raw_feature_names)
    group_indices = []
    for gname in grouped_names:
        indices = [i for i, f in enumerate(raw_feature_names)
                   if f == gname or f.startswith(gname + '_layer')]
        group_indices.append(indices)
    return grouped_names, group_indices


def _aggregate_product_measure_layers(data, group_indices):
    """Sum product measure data over grouped layer indices.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_eval_features, n_raw_model_features, x, y, z).
    group_indices : list of list of int
        Indices for each grouped feature.

    Returns
    -------
    np.ndarray
        Shape (n_eval_features, n_grouped_features, x, y, z).
    """
    grouped = []
    for indices in group_indices:
        grouped.append(np.nansum(data[:, indices, ...], axis=1))
    return np.stack(grouped, axis=1)


def _load_features_df(out_dir, model, sid, file_label):
    """Load the subset eval features CSV from known locations."""
    in_dir = os.path.join(out_dir, 'EncodingModel')
    group_dir = os.path.join(out_dir, 'SecondLevelGroup')

    # Try centralized features CSV first
    for path in [
        os.path.join(in_dir, 'features', f'encoding_model-{model}_subset_eval_features.csv'),
        os.path.join(group_dir, f'{sid}{file_label}_subset_eval_features.csv'),
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)

    raise FileNotFoundError(f"Could not find subset_eval_features.csv in {in_dir} or {group_dir}")


def plot_full_vs_subset_group_maps(
        base_dir, out_dir, figure_dir, model, smoothing_fwhm, mask, chunklen,
        model_feature_names, subset_eval_feature,
        population='NT', task='sherlock',
        features_df=None,
        threshold=0.00001, vmax=None,
        cmap='yellow_hot',
        ROI_niis=[], ROI_names=[], ROI_colors=[],
        mask_by_performance=True,
        group_layers=False, raw_model_feature_names=None):
    """Plot side-by-side brain maps: full time-series vs high-subset product measure.

    Creates a composite figure with rows = model features and two columns:
    column 1 = full evaluation, column 2 = high-subset evaluation for the
    specified eval feature.

    Parameters
    ----------
    base_dir, out_dir, figure_dir, model, smoothing_fwhm, mask, chunklen,
    population, task : standard pipeline params.
    model_feature_names : list of str
        Grouped model feature names to plot (e.g. ['alexnet_eps0.3', 'motion', ...]).
    subset_eval_feature : str
        Eval feature whose 'high' subset to compare (e.g. 'speaking').
    features_df : pd.DataFrame or None
        Eval features metadata. Loaded automatically if None.
    threshold, vmax, cmap : plotting params passed to plot_surface.
    ROI_niis, ROI_names, ROI_colors : ROI overlay params.
    mask_by_performance : bool
        Zero out voxels where full performance < 0.
    group_layers : bool
        If True, sum across layers for multi-layer models.
    raw_model_feature_names : list of str or None
        Raw (ungrouped) feature names, needed when group_layers=True.
    """
    from PIL import Image, ImageDraw, ImageFont
    import gc

    file_label = _build_file_label(model, smoothing_fwhm, mask, chunklen)
    sid = f'sub-{population}'
    analysis_dir = os.path.join(out_dir, 'SecondLevelGroup')

    # Load features metadata
    if features_df is None:
        features_df = _load_features_df(out_dir, model, sid, file_label)
    eval_feature_names = list(features_df['name'])

    # --- Load full product measure ---
    full_path = os.path.join(
        analysis_dir, 'ind_product_measure',
        f'{sid}{file_label}_measure-ind_product_measure_raw.nii.gz')
    full_nii = nibabel.load(full_path)
    full_data = full_nii.get_fdata()  # (n_raw_features, x, y, z)
    affine = full_nii.affine

    # Get raw feature names for indexing
    if raw_model_feature_names is None:
        raw_model_feature_names = _load_model_feature_names(out_dir, model, group_layers=False)

    # Build layer grouping if needed
    if group_layers:
        grouped_names = _load_model_feature_names(out_dir, model, group_layers=True)
        _, group_indices = _get_layer_grouping(raw_model_feature_names)
        # Only aggregate if data has raw (ungrouped) number of features
        if full_data.shape[0] == len(raw_model_feature_names):
            grouped_full = []
            for indices in group_indices:
                grouped_full.append(np.nansum(full_data[indices, ...], axis=0))
            full_data = np.stack(grouped_full, axis=0)
        # else: data is already grouped (shape matches grouped_names)
    else:
        grouped_names = raw_model_feature_names

    # Map model_feature_names to indices in grouped_names
    feat_indices = []
    for mf in model_feature_names:
        if mf in grouped_names:
            feat_indices.append(grouped_names.index(mf))
        else:
            print(f'WARNING: {mf} not found in feature names, skipping')
    if not feat_indices:
        print('ERROR: No matching model features found')
        return

    # --- Load subset high product measure ---
    try:
        subset_data, _ = load_subset_eval(
            analysis_dir, sid, file_label, 'ind_product_measure', 'high')
    except FileNotFoundError:
        print('Subset eval NIfTI not found, compiling from per-subject data...')
        compile_subset_eval_data(
            base_dir, out_dir, model, smoothing_fwhm, mask, chunklen,
            population=population, task=task, conditions=['high'])
        subset_data, _ = load_subset_eval(
            analysis_dir, sid, file_label, 'ind_product_measure', 'high')

    # subset_data shape: (n_eval_features, n_raw_model_features, x, y, z)
    # Only aggregate if data has raw (ungrouped) number of model features
    if group_layers and subset_data.shape[1] == len(raw_model_feature_names):
        subset_data = _aggregate_product_measure_layers(subset_data, group_indices)

    # Find eval feature index
    if subset_eval_feature not in eval_feature_names:
        print(f'ERROR: {subset_eval_feature} not in eval features: {eval_feature_names}')
        return
    eval_idx = eval_feature_names.index(subset_eval_feature)

    # Optional performance masking
    if mask_by_performance:
        try:
            perf_path = os.path.join(
                analysis_dir, 'performance',
                f'{sid}{file_label}_measure-perf_raw.nii.gz')
            perf_data = nibabel.load(perf_path).get_fdata()  # (x, y, z) or (n_features, x, y, z)
            if perf_data.ndim == 3:
                perf_mask = perf_data < 0
            else:
                # Use overall performance (sum across features)
                perf_mask = np.sum(perf_data, axis=0) < 0
        except FileNotFoundError:
            perf_mask = None
    else:
        perf_mask = None

    # --- Generate individual brain map PNGs ---
    tmp_dir = os.path.join(figure_dir, 'subset_eval', '_tmp_full_vs_subset')
    Path(tmp_dir).mkdir(exist_ok=True, parents=True)

    temp_files = []  # list of (full_png, subset_png) per model feature

    for mf, fi in zip(model_feature_names, feat_indices):
        # Full volume
        full_vol = full_data[fi].copy()
        if perf_mask is not None:
            full_vol[perf_mask] = 0
        full_nii_img = nibabel.Nifti1Image(full_vol, affine)
        full_filepath = os.path.join(tmp_dir, f'full_{mf}')
        plotting_helpers.plot_surface(
            full_nii_img, full_filepath,
            ROI_niis=ROI_niis, threshold=threshold, vmax=vmax,
            cmap=cmap, symmetric_cbar=False,
            colorbar_label='Explained Variance $R^2$',
            ROIs=ROI_names, ROI_colors=ROI_colors)

        # Subset high volume
        subset_vol = subset_data[eval_idx, fi].copy()
        if perf_mask is not None:
            subset_vol[perf_mask] = 0
        subset_nii_img = nibabel.Nifti1Image(subset_vol, affine)
        subset_filepath = os.path.join(tmp_dir, f'subset_{mf}')
        plotting_helpers.plot_surface(
            subset_nii_img, subset_filepath,
            ROI_niis=ROI_niis, threshold=threshold, vmax=vmax,
            cmap=cmap, symmetric_cbar=False,
            colorbar_label='Explained Variance $R^2$',
            ROIs=ROI_names, ROI_colors=ROI_colors)

        temp_files.append((full_filepath + '.png', subset_filepath + '.png'))
        gc.collect()

    # --- Compose grid figure with PIL ---
    n_rows = len(temp_files)
    images = []
    for full_png, subset_png in temp_files:
        images.append((Image.open(full_png), Image.open(subset_png)))

    # Get dimensions from first image
    cell_w, cell_h = images[0][0].size
    label_w = 200  # width for row labels
    header_h = 80  # height for column headers
    padding = 10

    total_w = label_w + 2 * cell_w + 3 * padding
    total_h = header_h + n_rows * (cell_h + padding) + padding

    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial.ttf", 40)
        header_font = ImageFont.truetype("Arial.ttf", 44)
    except IOError:
        font = ImageFont.load_default()
        header_font = font

    # Column headers
    col1_x = label_w + padding + cell_w // 2
    col2_x = label_w + 2 * padding + cell_w + cell_w // 2
    draw.text((col1_x, 10), 'Full', fill='black', font=header_font, anchor='mt')
    draw.text((col2_x, 10), f'High {subset_eval_feature}', fill='black',
              font=header_font, anchor='mt')

    # Rows
    for row_idx, ((full_img, subset_img), mf) in enumerate(zip(images, model_feature_names)):
        y = header_h + row_idx * (cell_h + padding) + padding
        # Row label
        label = MODEL_FEATURE_LABELS.get(mf, mf)
        draw.text((label_w - 10, y + cell_h // 2), label, fill='black',
                  font=font, anchor='rm')
        # Full map
        canvas.paste(full_img, (label_w + padding, y))
        # Subset map
        canvas.paste(subset_img, (label_w + 2 * padding + cell_w, y))

    # Save
    out_dir_fig = os.path.join(figure_dir, 'subset_eval')
    Path(out_dir_fig).mkdir(exist_ok=True, parents=True)
    output_path = os.path.join(
        out_dir_fig,
        f'{sid}{file_label}_full_vs_high_{subset_eval_feature}_group_maps.png')
    canvas.save(output_path, dpi=(300, 300))
    print(f'saved: {output_path}')

    # Cleanup temp files
    for full_img, subset_img in images:
        full_img.close()
        subset_img.close()
    for full_png, subset_png in temp_files:
        if os.path.exists(full_png):
            os.remove(full_png)
        if os.path.exists(subset_png):
            os.remove(subset_png)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return output_path


def run_group_maps(base_dir, out_dir, figure_dir, model, smoothing_fwhm, mask, chunklen,
                   population='NT', task='sherlock', compile_group=True, load=False,
                   conditions=None, model_feature_names=None, group_layers=False,
                   plot_perf=True, plot_product_measure=True,
                   roi_names=None,
                   threshold=0.00001, vmax_high_low=None, vmax_contrast=None,
                   cmap_positive='yellow_hot', cmap_contrast='blue_neg_yellow_pos',
                   mask_by_performance=True):
    """Compile group data and plot whole-brain surface maps.

    Parameters
    ----------
    base_dir : str
        Top-level project directory.
    out_dir : str
        Analysis output directory (parent of EncodingModel/, SecondLevelGroup/).
    figure_dir : str
        Figure output directory.
    model : str
        Encoding model name.
    smoothing_fwhm : float
        Smoothing FWHM.
    mask : str
        Mask name.
    chunklen : int
        Chunk length.
    population : str
        'NT'.
    task : str
        Task name.
    compile_group : bool
        If True, compile per-subject data into group averages first.
    load : bool
        If True, skip compilation and load existing group-level files.
    conditions : list of str or None
        Subset of ['high', 'low', 'contrast']. None for all.
    model_feature_names : list of str or None
        Model feature names for product measure plotting.
    plot_perf : bool
        Whether to plot performance surface maps.
    plot_product_measure : bool
        Whether to plot product measure surface maps.
    roi_names : list of str or None
        ROI names to overlay on brain maps (e.g. ['MT', 'STS', 'pTemp']).
    threshold, vmax_high_low, vmax_contrast, cmap_positive, cmap_contrast, mask_by_performance
        Plotting parameters.

    Returns
    -------
    features_df : pd.DataFrame
    """
    file_label = _build_file_label(model, smoothing_fwhm, mask, chunklen)
    sid = f'sub-{population}'
    group_dir = os.path.join(out_dir, 'SecondLevelGroup')
    fig_dir = os.path.join(figure_dir, 'subset_eval')

    # Auto-load model feature names if not provided
    raw_model_feature_names = None
    if model_feature_names is None and plot_product_measure:
        raw_model_feature_names = _load_model_feature_names(out_dir, model, group_layers=False)
        if group_layers:
            model_feature_names = _load_model_feature_names(out_dir, model, group_layers=True)
        else:
            model_feature_names = raw_model_feature_names
        print(f'Auto-loaded model feature names: {model_feature_names}')

    # Compile group-level data
    if compile_group:
        features_df = compile_subset_eval_data(
            base_dir, out_dir, model, smoothing_fwhm, mask, chunklen,
            population=population, task=task, conditions=conditions, load=load)
    else:
        features_df = _load_features_df(out_dir, model, sid, file_label)

    # Load ROI masks for overlay
    roi_niis, roi_colors_list = [], []
    if roi_names is not None:
        _, roi_niis, roi_colors_list = _load_roi_masks(base_dir, roi_names)

    plot_kwargs = dict(threshold=threshold, vmax_high_low=vmax_high_low,
                       vmax_contrast=vmax_contrast, cmap_positive=cmap_positive,
                       cmap_contrast=cmap_contrast)

    if plot_perf:
        plot_subset_eval_performance(
            group_dir, fig_dir, sid, file_label, features_df,
            conditions=conditions,
            ROI_niis=roi_niis, ROI_names=roi_names or [], ROI_colors=roi_colors_list,
            **plot_kwargs)

    if plot_product_measure and model_feature_names is not None:
        plot_subset_eval_product_measure(
            group_dir, fig_dir, sid, file_label, features_df,
            model_feature_names=model_feature_names,
            conditions=conditions,
            ROI_niis=roi_niis, ROI_names=roi_names or [], ROI_colors=roi_colors_list,
            mask_by_performance=mask_by_performance,
            group_layers=group_layers, raw_model_feature_names=raw_model_feature_names,
            **plot_kwargs)

    print('Done: run_group_maps()')
    return features_df


def extract_subset_eval_in_rois(base_dir, out_dir, model, smoothing_fwhm, mask, chunklen,
                                roi_names, population='NT', task='sherlock',
                                conditions=None, measure='perf',
                                model_feature_names=None, mask_by_performance=True,
                                use_glm_voxel_selection=False, perc_top_voxels=5,
                                glm_localizer_contrast=None, hemispheres=None,
                                group_layers=False, raw_model_feature_names=None):
    """Extract mean subset eval values within ROIs, returning a tidy DataFrame.

    Supports two modes:
    - Anatomical ROI (default): mean over all voxels in the anatomical mask.
    - GLM voxel selection: mean over top-N% voxels within the anatomical mask,
      selected by GLM localizer z-scores (per-subject).

    Parameters
    ----------
    base_dir : str
        Top-level project directory.
    out_dir : str
        Analysis output directory.
    model : str
        Encoding model name.
    smoothing_fwhm : float
        Smoothing FWHM.
    mask : str
        Encoding mask name (e.g. 'ISC').
    chunklen : int
        Chunk length.
    roi_names : list of str
        ROI names (e.g. ['MT', 'pSTS', 'aSTS', 'pTemp', 'aTemp']).
    population : str
        'NT'.
    task : str
        Task name.
    conditions : list of str or None
        Subset of CONDITIONS. None for all.
    measure : str
        'perf' or 'ind_product_measure'.
    model_feature_names : list of str or None
        Required if measure='ind_product_measure'.
    mask_by_performance : bool
        If True and measure='ind_product_measure', zero out voxels with perf < 0.
    use_glm_voxel_selection : bool
        If True, use GLM z-scores to select top voxels within each ROI per subject.
    perc_top_voxels : int
        Percentage for GLM voxel selection (e.g. 5 for top 5%).
    glm_localizer_contrast : dict or None
        ROI name -> (task, run, contrast) tuple or contrast string.
        E.g. {'MT': ('SIpointlights', '123', 'interact&no_interact'),
               'pSTS': ('SIpointlights', '123', 'interact-no_interact'),
               'pTemp': ('language', '12', 'intact-degraded')}
        If a plain string is given, defaults to task='SIpointlights', run='123'.
    hemispheres : list of str or None
        Hemispheres to evaluate separately, e.g. ['left', 'right']. None for bilateral.

    Returns
    -------
    results_df : pd.DataFrame
        Tidy dataframe with columns: eval_feature, condition, roi, hemisphere,
        mean_value, [model_feature if applicable], [subject if per-subject].
    """
    import nilearn.image

    file_label = _build_file_label(model, smoothing_fwhm, mask, chunklen)
    sid = f'sub-{population}'
    group_dir = os.path.join(out_dir, 'SecondLevelGroup')

    features_df = _load_features_df(out_dir, model, sid, file_label)
    eval_feature_names = list(features_df['name'])

    # Validate model_feature_names against actual model features
    if model_feature_names is not None and measure == 'ind_product_measure':
        actual_names = _load_model_feature_names(out_dir, model, group_layers=group_layers)
        for name in model_feature_names:
            if name not in actual_names:
                raise ValueError(
                    f"model_feature_names contains '{name}' which is not in the model's features: "
                    f"{actual_names}")

    # Build layer grouping if needed
    if group_layers and raw_model_feature_names is not None and measure == 'ind_product_measure':
        _, group_indices = _get_layer_grouping(raw_model_feature_names)
    else:
        group_indices = None

    if conditions is None:
        conditions = CONDITIONS
    if hemispheres is None:
        hemispheres = ['both']

    loader = Namespace(dir=base_dir, brain_shape=BRAIN_SHAPE)

    rows = []

    if use_glm_voxel_selection:
        # Per-subject extraction with GLM voxel selection
        subjects = helpers.get_subjects(population)[task]
        file_label_new = _build_file_label(model, smoothing_fwhm, mask, chunklen)
        file_label_old = _build_file_label_old(model, smoothing_fwhm, mask, chunklen)
        in_dir = os.path.join(out_dir, 'EncodingModel')

        n_voxels_all = {
            '1': {'MT': 34, 'STS': 85, 'temporal_language': 85, 'frontal': 32, 'pSTS': 60, 'aSTS': 25, 'pTemp': 60, 'aTemp': 25},
            '2.5': {'MT': 86, 'STS': 212, 'temporal_language': 212, 'frontal': 79, 'pSTS': 150, 'aSTS': 62, 'pTemp': 150, 'aTemp': 62},
            '5': {'MT': 172, 'STS': 425, 'temporal_language': 425, 'frontal': 158, 'pSTS': 300, 'aSTS': 125, 'pTemp': 300, 'aTemp': 125},
            '7.5': {'MT': 259, 'STS': 637, 'temporal_language': 637, 'frontal': 236, 'pSTS': 450, 'aSTS': 187, 'pTemp': 450, 'aTemp': 187},
            '10': {'MT': 345, 'STS': 850, 'temporal_language': 850, 'frontal': 315, 'pSTS': 600, 'aSTS': 250, 'pTemp': 600, 'aTemp': 250},
            '12.5': {'MT': 432, 'STS': 1062, 'temporal_language': 1062, 'frontal': 394, 'pSTS': 750, 'aSTS': 312, 'pTemp': 750, 'aTemp': 312},
            '15': {'MT': 518, 'STS': 1275, 'temporal_language': 1275, 'frontal': 473, 'pSTS': 900, 'aSTS': 375, 'pTemp': 900, 'aTemp': 375},
            '20': {'MT': 690, 'STS': 1700, 'temporal_language': 1700, 'frontal': 630, 'pSTS': 1200, 'aSTS': 500, 'pTemp': 1200, 'aTemp': 500},
        }
        perc_key = str(perc_top_voxels)

        for subject in subjects:
            # Load per-subject GLM z-scores for localizer
            for condition in conditions:
                for roi_name in roi_names:
                    for hemi in hemispheres:
                        hemi_roi = f'{hemi}-{roi_name}' if hemi != 'both' else roi_name
                        roi_mask_nii = helpers.load_mask(loader, hemi_roi)
                        roi_mask = roi_mask_nii.get_fdata()

                        # Load GLM z-scores to select top voxels
                        if glm_localizer_contrast is not None and roi_name in glm_localizer_contrast:
                            loc_entry = glm_localizer_contrast[roi_name]
                            if isinstance(loc_entry, tuple):
                                glm_task, glm_run, contrast = loc_entry
                            else:
                                # Legacy: just a contrast string, default to SIpointlights/123
                                glm_task, glm_run, contrast = 'SIpointlights', '123', loc_entry
                            glm_dir = os.path.join(out_dir, 'GLM', subject)
                            space = 'MNI152NLin2009cAsym'
                            glm_file = os.path.join(
                                glm_dir,
                                f'{subject}_task-{glm_task}_space-{space}_run-{glm_run}_measure-zscore_contrast-{contrast}.nii.gz')
                            if os.path.exists(glm_file):
                                z_img = nibabel.load(glm_file)
                                z_data = z_img.get_fdata()
                                # Resample roi_mask to z_data shape if needed
                                if roi_mask.shape != z_data.shape:
                                    roi_mask_nii = nilearn.image.resample_img(
                                        roi_mask_nii, target_affine=z_img.affine,
                                        target_shape=z_data.shape, interpolation='nearest')
                                    roi_mask = roi_mask_nii.get_fdata()

                                # Select top N voxels within ROI
                                in_roi = roi_mask > 0
                                z_in_roi = z_data[in_roi]
                                n_target = n_voxels_all.get(perc_key, {}).get(roi_name, len(z_in_roi))
                                n_target = min(n_target, len(z_in_roi))
                                if n_target > 0 and len(z_in_roi) > 0:
                                    sorted_z = np.sort(z_in_roi)[::-1]
                                    threshold_val = sorted_z[min(n_target - 1, len(sorted_z) - 1)]
                                    selection_mask = in_roi & (z_data >= threshold_val)
                                else:
                                    selection_mask = in_roi
                                print(f'  {subject} {roi_name} {hemi}: '
                                      f'{np.sum(selection_mask)}/{np.sum(in_roi)} voxels selected '
                                      f'(target {n_target})')
                            else:
                                print(f'WARNING: GLM file not found for {subject} {roi_name}, '
                                      f'skipping: {glm_file}')
                                continue
                        else:
                            print(f'WARNING: No GLM contrast for ROI {roi_name}, skipping')
                            continue

                        # Load per-subject subset eval data
                        try:
                            measure_label = f'perf_subset_{condition}_eval_raw' if measure == 'perf' \
                                else f'ind_product_measure_subset_{condition}_eval_raw'
                            nii = _load_flexible(in_dir,
                                                 'performance' if measure == 'perf' else 'ind_product_measure',
                                                 subject, file_label_new, file_label_old, measure_label)
                            data = nii.get_fdata()
                        except FileNotFoundError:
                            continue

                        # Aggregate layers if grouping
                        if group_indices is not None:
                            data = _aggregate_product_measure_layers(data, group_indices)

                        # Resample selection_mask to data shape if needed
                        if selection_mask.shape != data.shape[-3:]:
                            sel_nii = nibabel.Nifti1Image(selection_mask.astype('float32'), roi_mask_nii.affine)
                            sel_nii = nilearn.image.resample_img(
                                sel_nii, target_affine=nii.affine,
                                target_shape=data.shape[-3:], interpolation='nearest')
                            selection_mask = sel_nii.get_fdata() > 0

                        # Optionally load perf for masking
                        perf_mask = None
                        if mask_by_performance and measure == 'ind_product_measure':
                            try:
                                perf_nii = _load_flexible(
                                    in_dir, 'performance', subject,
                                    file_label_new, file_label_old,
                                    f'perf_subset_{condition}_eval_raw')
                                perf_mask = perf_nii.get_fdata()
                            except FileNotFoundError:
                                pass

                        for f_idx, eval_feat in enumerate(eval_feature_names):
                            if measure == 'perf':
                                vol = data[f_idx]
                                vals = vol[selection_mask]
                                rows.append({
                                    'subject': subject, 'eval_feature': eval_feat,
                                    'condition': condition, 'roi': roi_name,
                                    'hemisphere': hemi, 'mean_value': np.nanmean(vals)
                                })
                            else:
                                for m_idx, model_feat in enumerate(model_feature_names):
                                    vol = data[f_idx, m_idx].copy()
                                    if perf_mask is not None:
                                        vol[perf_mask[f_idx] < 0] = 0
                                    vals = vol[selection_mask]
                                    rows.append({
                                        'subject': subject, 'eval_feature': eval_feat,
                                        'condition': condition, 'roi': roi_name,
                                        'hemisphere': hemi, 'model_feature': model_feat,
                                        'mean_value': np.nanmean(vals)
                                    })
    else:
        # Per-subject anatomical ROI extraction
        subjects = helpers.get_subjects(population)[task]
        file_label_new = _build_file_label(model, smoothing_fwhm, mask, chunklen)
        file_label_old = _build_file_label_old(model, smoothing_fwhm, mask, chunklen)
        in_dir = os.path.join(out_dir, 'EncodingModel')

        for subject in subjects:
            for condition in conditions:
                measure_label = f'perf_subset_{condition}_eval_raw' if measure == 'perf' \
                    else f'ind_product_measure_subset_{condition}_eval_raw'
                try:
                    nii = _load_flexible(in_dir,
                                         'performance' if measure == 'perf' else 'ind_product_measure',
                                         subject, file_label_new, file_label_old, measure_label)
                    data = nii.get_fdata()
                except FileNotFoundError:
                    continue

                # Aggregate layers if grouping
                if group_indices is not None:
                    data = _aggregate_product_measure_layers(data, group_indices)

                # Optionally load perf for masking
                perf_data = None
                if mask_by_performance and measure == 'ind_product_measure':
                    try:
                        perf_nii = _load_flexible(
                            in_dir, 'performance', subject,
                            file_label_new, file_label_old,
                            f'perf_subset_{condition}_eval_raw')
                        perf_data = perf_nii.get_fdata()
                    except FileNotFoundError:
                        pass

                for roi_name in roi_names:
                    for hemi in hemispheres:
                        hemi_roi = f'{hemi}-{roi_name}' if hemi != 'both' else roi_name
                        roi_mask_nii = helpers.load_mask(loader, hemi_roi)
                        # Resample to data space
                        roi_mask_nii = nilearn.image.resample_img(
                            roi_mask_nii, target_affine=nii.affine,
                            target_shape=data.shape[-3:], interpolation='nearest')
                        roi_mask_bool = roi_mask_nii.get_fdata() > 0

                        for f_idx, eval_feat in enumerate(eval_feature_names):
                            if measure == 'perf':
                                vals = data[f_idx][roi_mask_bool]
                                rows.append({
                                    'subject': subject, 'eval_feature': eval_feat,
                                    'condition': condition, 'roi': roi_name,
                                    'hemisphere': hemi, 'mean_value': np.nanmean(vals)
                                })
                            else:
                                for m_idx, model_feat in enumerate(model_feature_names):
                                    vol = data[f_idx, m_idx].copy()
                                    if perf_data is not None:
                                        vol[perf_data[f_idx] < 0] = 0
                                    vals = vol[roi_mask_bool]
                                    rows.append({
                                        'subject': subject, 'eval_feature': eval_feat,
                                        'condition': condition, 'roi': roi_name,
                                        'hemisphere': hemi, 'model_feature': model_feat,
                                        'mean_value': np.nanmean(vals)
                                    })

    return pd.DataFrame(rows)


def _merge_anterior_posterior_df(df, roi_names, merge_pairs=None):
    """Merge anterior/posterior ROI pairs by averaging per subject.

    Returns (merged_df, new_roi_names).
    """
    if merge_pairs is None:
        merge_pairs = {'STS': ['pSTS', 'aSTS'], 'Temp': ['pTemp', 'aTemp']}

    merged_rows = []
    for merged_name, sub_rois in merge_pairs.items():
        sub_df = df[df['roi'].isin(sub_rois)]
        if sub_df.empty:
            continue
        group_cols = [c for c in sub_df.columns if c not in ('roi', 'mean_value')]
        avg_df = sub_df.groupby(group_cols, as_index=False)['mean_value'].mean()
        avg_df['roi'] = merged_name
        merged_rows.append(avg_df)

    all_sub_rois = [r for pair in merge_pairs.values() for r in pair]
    df = df[~df['roi'].isin(all_sub_rois)]
    if merged_rows:
        df = pd.concat([df] + merged_rows, ignore_index=True)

    # Rebuild roi_names
    new_roi_names = []
    used_merges = set()
    for r in roi_names:
        replaced = False
        for merged_name, sub_rois in merge_pairs.items():
            if r in sub_rois and merged_name not in used_merges:
                new_roi_names.append(merged_name)
                used_merges.add(merged_name)
                replaced = True
                break
            elif r in sub_rois:
                replaced = True
                break
        if not replaced:
            new_roi_names.append(r)

    return df, new_roi_names


def _plot_roi_stacked_bars(df, roi_names, fig_dir, sid, file_label, use_glm=False,
                           features_info=None, split_hemi=False):
    """Create stacked barplots: one figure per eval_feature, ROIs on x-axis,
    high/low as paired bars with model features stacked within each bar.

    Parameters
    ----------
    features_info : pd.DataFrame or None
        DataFrame with columns 'name', 'high_TRs', 'low_TRs' from the
        subset_eval_features CSV. Used to annotate titles with split sizes.
    split_hemi : bool
        If True, plot left and right hemispheres as separate x-axis entries.
        If False (default), average across hemispheres.
    """
    if df.empty:
        return

    glm_tag = '_glm' if use_glm else ''
    has_hemi = (split_hemi and 'hemisphere' in df.columns
                and set(df['hemisphere'].unique()) != {'both'})

    # Filter to high/low only
    df = df[df['condition'].isin(['high', 'low'])].copy()

    # Average across hemispheres if not splitting
    if not has_hemi and 'hemisphere' in df.columns and set(df['hemisphere'].unique()) != {'both'}:
        group_cols = [c for c in df.columns if c not in ('hemisphere', 'mean_value')]
        df = df.groupby(group_cols, as_index=False)['mean_value'].mean()

    # Build x-axis labels: split by hemisphere if requested
    if has_hemi:
        hemis = sorted(df['hemisphere'].unique())
        x_labels = []
        for roi in roi_names:
            for hemi in hemis:
                x_labels.append(f'{hemi} {roi}')
    else:
        x_labels = list(roi_names)

    eval_features = df['eval_feature'].unique()
    model_features = df['model_feature'].unique()

    # Use colors_dict from plotting_helpers for model feature colors
    colors_dict = plotting_helpers.get_colors_dict()
    feature_colors = {}
    cmap_fallback = plt.cm.get_cmap('Set2', len(model_features))
    for i, mf in enumerate(model_features):
        feature_colors[mf] = colors_dict.get(mf, cmap_fallback(i))

    # Build split-size lookup from features_info
    split_info = {}
    if features_info is not None:
        for _, row in features_info.iterrows():
            split_info[row['name']] = (int(row['high_TRs']), int(row['low_TRs']))

    bar_width = 0.3

    for eval_feat in eval_features:
        ef_df = df[df['eval_feature'] == eval_feat]

        fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(x_labels)), 5))

        x = np.arange(len(x_labels))

        for c_idx, cond in enumerate(['high', 'low']):
            cond_df = ef_df[ef_df['condition'] == cond]
            bottoms = np.zeros(len(x_labels))

            for mf_idx, mf in enumerate(model_features):
                means = []
                for roi_name in roi_names:
                    if has_hemi:
                        for hemi in hemis:
                            subset = cond_df[(cond_df['roi'] == roi_name) &
                                             (cond_df['hemisphere'] == hemi) &
                                             (cond_df['model_feature'] == mf)]
                            if 'subject' in subset.columns and len(subset) > 1:
                                means.append(subset['mean_value'].mean())
                            else:
                                means.append(subset['mean_value'].mean() if len(subset) > 0 else 0)
                    else:
                        subset = cond_df[(cond_df['roi'] == roi_name) &
                                         (cond_df['model_feature'] == mf)]
                        if 'subject' in subset.columns and len(subset) > 1:
                            means.append(subset['mean_value'].mean())
                        else:
                            means.append(subset['mean_value'].mean() if len(subset) > 0 else 0)

                means = np.clip(np.array(means), 0, None)

                ax.bar(x + c_idx * bar_width, means, bar_width,
                       bottom=bottoms,
                       label=mf if c_idx == 0 else None,
                       color=feature_colors[mf],
                       alpha=0.85,
                       edgecolor='white', linewidth=0.5,
                       hatch='/' if cond == 'low' else None)

                bottoms += means

        # Title with split sizes
        if eval_feat in split_info:
            high_n, low_n = split_info[eval_feat]
            title = f'{eval_feat}  (high: {high_n} TRs, low: {low_n} TRs)'
        else:
            title = eval_feat
        ax.set_title(title, fontsize=14, fontweight='normal')
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(x_labels, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Mean product measure $R^2$', fontsize=12)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(False)
        ax.spines[['top', 'right']].set_visible(False)

        # Patch-based legend outside plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=feature_colors[mf], edgecolor='k',
                  label=MODEL_FEATURE_LABELS.get(mf, mf))
            for mf in model_features
        ]
        ax.legend(handles=legend_elements, loc='center left',
                  bbox_to_anchor=(1, 0.5), frameon=False, fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(
            fig_dir,
            f'{sid}{file_label}_ind_product_measure_roi_stacked_{eval_feat}{glm_tag}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'saved: {filepath}')


def plot_full_vs_subset_stacked_bars(subset_csv, full_csv, roi_names,
                                     model_feature_names, subset_eval_features,
                                     fig_dir, sid, file_label,
                                     features_info=None, split_hemi=False,
                                     merge_anterior_posterior=False, overlay_points=False,
                                     save_source_data=False):
    """Plot stacked barplots comparing full time-series vs high-subset product measure.

    For each specified subset eval feature, creates one figure with ROIs on the
    x-axis and two stacked bars per ROI: one for the full time-series product
    measure and one for the 'high' subset of that eval feature.

    Parameters
    ----------
    subset_csv : str
        Path to the subset_eval ROI product measure CSV (from run_roi_analysis).
        Columns: subject, eval_feature, condition, roi, hemisphere, model_feature,
        mean_value.
    full_csv : str
        Path to the full product measure GLM summary CSV (from
        SecondLevelIndividual).  Columns include: subject, mask (=ROI),
        hemisphere, enc_feature_name, encoding_response.
    roi_names : list of str
        ROI names to plot (must exist in both CSVs).
    model_feature_names : list of str
        Model features to stack (e.g. alexnet_eps0.3, motion, …).
    subset_eval_features : list of str
        Which eval features to create plots for (e.g. ['speaking']).
    fig_dir : str
        Output directory for figures.
    sid : str
        Subject/population identifier (e.g. 'sub-NT').
    file_label : str
        File label for output filenames.
    features_info : pd.DataFrame or None
        DataFrame with 'name', 'high_TRs', 'low_TRs', 'total_TRs' for title
        annotation.
    split_hemi : bool
        If True, plot hemispheres as separate x-axis entries.
    merge_anterior_posterior : bool
        If True, average pSTS+aSTS -> STS and pTemp+aTemp -> Temp.
    overlay_points : bool
        If True, also draw each participant's total (summed over the stacked
        model features) as a point on each bar.
    save_source_data : bool
        If True, save each participant's value for every bar segment (region,
        full movie or subset, model feature) as a csv next to the figure.

    When plotting_helpers.figure_width_mm is set, the figure is saved as an
    editable pdf of that width with plotting_helpers.figure_fontsize fonts.
    """
    publication = plotting_helpers.figure_width_mm is not None
    _fs = plotting_helpers._fs
    plotting_helpers._update_font_rc(10)
    if publication:
        plt.rcParams['hatch.linewidth'] = 0.5
    subset_df = pd.read_csv(subset_csv)
    full_df = pd.read_csv(full_csv)
    # use the all-runs localizer rows (the same voxels as the subset evaluation), one per participant/region/hemisphere/feature;
    # the cross-validated-split rows select different voxels
    full_df = full_df[full_df['averaged_localizer_runs'].notna()]  # empty placeholder rows (no runs)
    full_df = full_df[[len(str(runs).split(',')) == 1 for runs in full_df['averaged_localizer_runs']]]
    full_df = full_df.drop_duplicates(subset=['subject', 'mask', 'hemisphere', 'enc_feature_name'])

    # Map full CSV column names to common names and keep only needed columns
    full_df = full_df.rename(columns={'mask': 'roi', 'enc_feature_name': 'model_feature',
                                      'encoding_response': 'mean_value'})
    full_df = full_df[['subject', 'roi', 'hemisphere', 'model_feature', 'mean_value']]
    # Keep only the model features we need
    full_df = full_df[full_df['model_feature'].isin(model_feature_names)]
    full_df = full_df[full_df['roi'].isin(roi_names)]

    # Merge anterior/posterior ROIs (must use original roi_names for both)
    if merge_anterior_posterior:
        orig_roi_names = list(roi_names)
        full_df, roi_names = _merge_anterior_posterior_df(full_df, orig_roi_names)
        subset_df, _ = _merge_anterior_posterior_df(subset_df, orig_roi_names)

    # Determine hemisphere handling
    has_hemi_full = 'hemisphere' in full_df.columns and set(full_df['hemisphere'].unique()) != {'both'}
    has_hemi_sub = 'hemisphere' in subset_df.columns and set(subset_df['hemisphere'].unique()) != {'both'}
    has_hemi = split_hemi and (has_hemi_full or has_hemi_sub)

    if has_hemi:
        hemis = sorted(set(
            list(full_df['hemisphere'].unique() if has_hemi_full else []) +
            list(subset_df['hemisphere'].unique() if has_hemi_sub else [])
        ) - {'both'})
        x_labels = []
        for roi in roi_names:
            for hemi in hemis:
                x_labels.append(f'{hemi} {roi}')
    else:
        x_labels = list(roi_names)
        # Average across hemispheres
        full_df = full_df.groupby(['subject', 'roi', 'model_feature'],
                                  as_index=False)['mean_value'].mean()

    colors_dict = plotting_helpers.get_colors_dict()
    cmap_fallback = plt.cm.get_cmap('Set2', len(model_feature_names))
    feature_colors = {}
    for i, mf in enumerate(model_feature_names):
        feature_colors[mf] = colors_dict.get(mf, cmap_fallback(i))

    # Build split-size lookup
    split_info = {}
    if features_info is not None:
        for _, row in features_info.iterrows():
            split_info[row['name']] = {
                'high': int(row['high_TRs']),
                'low': int(row['low_TRs']),
                'total': int(row['total_TRs']),
            }

    bar_width = 0.3
    Path(fig_dir).mkdir(exist_ok=True, parents=True)

    for eval_feat in subset_eval_features:
        # Get the 'high' subset data for this eval feature
        high_df = subset_df[(subset_df['eval_feature'] == eval_feat) &
                            (subset_df['condition'] == 'high')].copy()
        if high_df.empty:
            print(f'WARNING: No high-subset data for {eval_feat}, skipping.')
            continue

        # Average across hemispheres only if not splitting
        if not has_hemi and 'hemisphere' in high_df.columns:
            high_df = high_df.groupby(['subject', 'roi', 'model_feature'],
                                      as_index=False)['mean_value'].mean()

        fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(x_labels)), 5))
        x = np.arange(len(x_labels))

        bar_labels = ['full', f'high {eval_feat}']
        bar_dfs = [full_df, high_df]
        hatch_patterns = [None, '///' if publication else '/']

        for b_idx, (label, b_df, hatch) in enumerate(
                zip(bar_labels, bar_dfs, hatch_patterns)):
            bottoms = np.zeros(len(x_labels))
            for mf_idx, mf in enumerate(model_feature_names):
                means = []
                for roi_name in roi_names:
                    if has_hemi:
                        for hemi in hemis:
                            subset = b_df[(b_df['roi'] == roi_name) &
                                          (b_df['hemisphere'] == hemi) &
                                          (b_df['model_feature'] == mf)]
                            if 'subject' in subset.columns and len(subset) > 1:
                                means.append(subset['mean_value'].mean())
                            else:
                                means.append(subset['mean_value'].mean()
                                             if len(subset) > 0 else 0)
                    else:
                        subset = b_df[(b_df['roi'] == roi_name) &
                                      (b_df['model_feature'] == mf)]
                        if 'subject' in subset.columns and len(subset) > 1:
                            means.append(subset['mean_value'].mean())
                        else:
                            means.append(subset['mean_value'].mean()
                                         if len(subset) > 0 else 0)

                means = np.clip(np.array(means), 0, None)

                ax.bar(x + b_idx * bar_width, means, bar_width,
                       bottom=bottoms,
                       label=mf if b_idx == 0 else None,
                       color=feature_colors[mf], alpha=0.85,
                       edgecolor='white', linewidth=plotting_helpers._lw(0.5) / 2 if publication else 0.5,
                       hatch=hatch)
                bottoms += means

            if overlay_points:
                # each participant's total across the stacked model features
                group_cols = ['roi', 'hemisphere'] if has_hemi else ['roi']
                totals = (b_df[b_df['model_feature'].isin(model_feature_names)]
                          .groupby(['subject'] + group_cols)['mean_value'].sum().reset_index())
                rng = np.random.default_rng(0)
                for x_idx, x_label in enumerate(x_labels):
                    if has_hemi:
                        hemi, roi_name = x_label.split(' ', 1)
                        values = totals[(totals['roi'] == roi_name) & (totals['hemisphere'] == hemi)]['mean_value']
                    else:
                        values = totals[totals['roi'] == x_label]['mean_value']
                    jitter = rng.uniform(-0.25, 0.25, len(values)) * bar_width
                    ax.scatter(x_idx + b_idx * bar_width + jitter, values, s=1.5 if publication else 8,
                               color='black', alpha=0.5, linewidths=0, zorder=4)

        # Title with TR counts
        if eval_feat in split_info:
            info = split_info[eval_feat]
            title = (f'Full Movie ({info["total"]} TRs) vs '
                     f'{eval_feat} present ({info["high"]} TRs)')
        else:
            title = f'Full vs {eval_feat} present'
        ax.set_title(title, fontsize=_fs(14), fontweight='normal')
        ax.set_xticks(x + bar_width / 2)
        if publication:
            ax.set_xticklabels(x_labels)
        else:
            ax.set_xticklabels(x_labels, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Cumulative explained variance $R^2$', fontsize=_fs(12))
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(False)
        ax.spines[['top', 'right']].set_visible(False)

        # Patch-based legend outside plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=feature_colors[mf], edgecolor='k',
                  linewidth=plotting_helpers._lw(1),
                  label=MODEL_FEATURE_LABELS.get(mf, mf))
            for mf in model_feature_names
        ]
        # the two bars of each region: unhatched = full movie, hatched = only the subset
        legend_elements += [
            Patch(facecolor='white', edgecolor='k', linewidth=plotting_helpers._lw(1), label='full movie'),
            Patch(facecolor='white', edgecolor='k', linewidth=plotting_helpers._lw(1), hatch='///', label=f'{eval_feat} present'),
        ] if publication else []
        ax.legend(handles=legend_elements, loc='center left',
                  bbox_to_anchor=(1, 0.5), frameon=False, fontsize=_fs(9))

        plt.tight_layout()
        filepath = os.path.join(
            fig_dir,
            f'{sid}{file_label}_ind_product_measure_full_vs_high_{eval_feat}')
        if publication:
            filepath += '.pdf'
            plotting_helpers.save_figure(fig, filepath)
        else:
            filepath += '.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'saved: {filepath}')

        if save_source_data:
            source_df = pd.concat([b_df.assign(timepoints=label) for label, b_df in
                                   zip(['full movie', f'{eval_feat} present'], bar_dfs)], ignore_index=True)
            source_df = source_df[source_df['model_feature'].isin(model_feature_names)]
            keep = ['subject', 'timepoints', 'roi'] + (['hemisphere'] if has_hemi else []) + ['model_feature', 'mean_value']
            source_df = source_df[keep].rename(columns={'roi': 'region', 'model_feature': 'feature',
                                                        'mean_value': 'product_measure'})
            source_df['feature'] = source_df['feature'].map(lambda mf: MODEL_FEATURE_LABELS.get(mf, mf))
            source_path = os.path.splitext(filepath)[0] + '_source_data.csv'
            source_df.to_csv(source_path, index=False)
            print(f'saved: {source_path}')


def run_roi_analysis(base_dir, out_dir, figure_dir, model, smoothing_fwhm, mask, chunklen,
                     roi_names, population='NT', task='sherlock',
                     conditions=None, model_feature_names=None, group_layers=False,
                     plot_perf=True, plot_product_measure=True,
                     mask_by_performance=True,
                     use_glm_voxel_selection=False, perc_top_voxels=5,
                     glm_localizer_contrast=None, hemispheres=None,
                     load=False, merge_anterior_posterior=False):
    """Extract subset eval values in ROIs and create bar plots.

    Parameters
    ----------
    base_dir, out_dir, figure_dir, model, smoothing_fwhm, mask, chunklen,
    population, task : standard pipeline params.
    roi_names : list of str
        ROI names (e.g. ['MT', 'pSTS', 'aSTS', 'pTemp', 'aTemp']).
    conditions : list of str or None
        Subset of CONDITIONS. None for all.
    model_feature_names : list of str or None
        Required for product measure.
    plot_perf, plot_product_measure : bool
        Which measures to plot.
    mask_by_performance : bool
        Zero out negative-performance voxels for product measure.
    use_glm_voxel_selection : bool
        If True, select top voxels per subject using GLM z-scores.
    perc_top_voxels : int
        Top percent of voxels for GLM selection.
    glm_localizer_contrast : dict or None
        ROI name -> GLM contrast name mapping.
    hemispheres : list of str or None
        e.g. ['left', 'right']. None for bilateral.
    load : bool
        If True, load previously saved result CSVs instead of recomputing.
    merge_anterior_posterior : bool
        If True, average pSTS+aSTS -> STS and pTemp+aTemp -> Temp per subject
        before plotting.

    Returns
    -------
    results : dict
        'perf_df' and/or 'product_measure_df' DataFrames.
    """
    file_label = _build_file_label(model, smoothing_fwhm, mask, chunklen)
    sid = f'sub-{population}'
    fig_dir = os.path.join(figure_dir, 'subset_eval', 'roi_summary')
    Path(fig_dir).mkdir(exist_ok=True, parents=True)

    # Load features info for stacked bar title annotations
    try:
        features_info = _load_features_df(out_dir, model, sid, file_label)
    except FileNotFoundError:
        features_info = None

    glm_tag = '_glm' if use_glm_voxel_selection else ''
    # per-participant ROI values, so with the other ROI analyses in SecondLevelIndividual
    csv_dir = os.path.join(out_dir, 'SecondLevelIndividual', 'subset_eval', 'roi_summary')
    Path(csv_dir).mkdir(exist_ok=True, parents=True)
    perf_csv = os.path.join(csv_dir, f'{sid}{file_label}_roi_perf{glm_tag}.csv')
    pm_csv = os.path.join(csv_dir, f'{sid}{file_label}_roi_product_measure{glm_tag}.csv')

    if load:
        results = {}
        if plot_perf:
            perf_df = pd.read_csv(perf_csv)
            # Filter by specified parameters
            if conditions is not None:
                perf_df = perf_df[perf_df['condition'].isin(conditions)]
            perf_df = perf_df[perf_df['roi'].isin(roi_names)]
            results['perf_df'] = perf_df
            print(f'Loaded perf_df from {perf_csv}')
            _plot_roi_bars(perf_df, roi_names, fig_dir, sid, file_label,
                           measure='perf', use_glm=use_glm_voxel_selection,
                           merge_anterior_posterior=merge_anterior_posterior)
        if plot_product_measure:
            pm_df = pd.read_csv(pm_csv)
            # Filter by specified parameters
            if conditions is not None:
                pm_df = pm_df[pm_df['condition'].isin(conditions)]
            pm_df = pm_df[pm_df['roi'].isin(roi_names)]
            if model_feature_names is not None:
                pm_df = pm_df[pm_df['model_feature'].isin(model_feature_names)]
            results['product_measure_df'] = pm_df
            print(f'Loaded product_measure_df from {pm_csv}')
            _plot_roi_bars(pm_df, roi_names, fig_dir, sid, file_label,
                           measure='ind_product_measure',
                           use_glm=use_glm_voxel_selection,
                           merge_anterior_posterior=merge_anterior_posterior)
            _plot_roi_stacked_bars(pm_df, roi_names, fig_dir, sid, file_label,
                                   use_glm=use_glm_voxel_selection,
                                   features_info=features_info)
        print('Done: run_roi_analysis() [loaded from CSV]')
        return results

    # Auto-load model feature names if not provided
    raw_model_feature_names = None
    if model_feature_names is None and plot_product_measure:
        raw_model_feature_names = _load_model_feature_names(out_dir, model, group_layers=False)
        if group_layers:
            model_feature_names = _load_model_feature_names(out_dir, model, group_layers=True)
        else:
            model_feature_names = raw_model_feature_names
        print(f'Auto-loaded model feature names: {model_feature_names}')

    results = {}

    if plot_perf:
        perf_df = extract_subset_eval_in_rois(
            base_dir, out_dir, model, smoothing_fwhm, mask, chunklen,
            roi_names, population=population, task=task,
            conditions=conditions, measure='perf',
            use_glm_voxel_selection=use_glm_voxel_selection,
            perc_top_voxels=perc_top_voxels,
            glm_localizer_contrast=glm_localizer_contrast,
            hemispheres=hemispheres)
        results['perf_df'] = perf_df
        perf_df.to_csv(perf_csv, index=False)

        _plot_roi_bars(perf_df, roi_names, fig_dir, sid, file_label,
                       measure='perf', use_glm=use_glm_voxel_selection,
                       merge_anterior_posterior=merge_anterior_posterior)

    if plot_product_measure and model_feature_names is not None:
        pm_df = extract_subset_eval_in_rois(
            base_dir, out_dir, model, smoothing_fwhm, mask, chunklen,
            roi_names, population=population, task=task,
            conditions=conditions, measure='ind_product_measure',
            model_feature_names=model_feature_names,
            mask_by_performance=mask_by_performance,
            use_glm_voxel_selection=use_glm_voxel_selection,
            perc_top_voxels=perc_top_voxels,
            glm_localizer_contrast=glm_localizer_contrast,
            hemispheres=hemispheres,
            group_layers=group_layers, raw_model_feature_names=raw_model_feature_names)
        results['product_measure_df'] = pm_df
        pm_df.to_csv(pm_csv, index=False)

        _plot_roi_bars(pm_df, roi_names, fig_dir, sid, file_label,
                       measure='ind_product_measure',
                       use_glm=use_glm_voxel_selection,
                       merge_anterior_posterior=merge_anterior_posterior)
        _plot_roi_stacked_bars(pm_df, roi_names, fig_dir, sid, file_label,
                               use_glm=use_glm_voxel_selection,
                               features_info=features_info)

    print('Done: run_roi_analysis()')
    return results


def _plot_roi_bars(df, roi_names, fig_dir, sid, file_label, measure='perf', use_glm=False,
                   merge_anterior_posterior=False):
    """Create bar plots from the tidy extraction DataFrame."""
    if df.empty:
        print(f'WARNING: No data to plot for {measure} (empty DataFrame). '
              f'Check that per-subject files and GLM localizer files exist.')
        return

    # Merge anterior/posterior ROIs by averaging per subject
    if merge_anterior_posterior:
        merge_pairs = {'STS': ['pSTS', 'aSTS'], 'Temp': ['pTemp', 'aTemp']}
        merged_rows = []
        for merged_name, sub_rois in merge_pairs.items():
            sub_df = df[df['roi'].isin(sub_rois)]
            if sub_df.empty:
                continue
            # Average across sub-ROIs per subject (and per model_feature if present)
            group_cols = ['subject', 'eval_feature', 'condition']
            if 'hemisphere' in sub_df.columns:
                group_cols.append('hemisphere')
            if 'model_feature' in sub_df.columns:
                group_cols.append('model_feature')
            avg_df = sub_df.groupby(group_cols, as_index=False)['mean_value'].mean()
            avg_df['roi'] = merged_name
            merged_rows.append(avg_df)

        # Remove the original sub-ROI rows, add merged
        all_sub_rois = [r for pair in merge_pairs.values() for r in pair]
        df = df[~df['roi'].isin(all_sub_rois)]
        if merged_rows:
            df = pd.concat([df] + merged_rows, ignore_index=True)

        # Rebuild roi_names: replace first occurrence of a sub-ROI with merged name
        new_roi_names = []
        used_merges = set()
        for r in roi_names:
            replaced = False
            for merged_name, sub_rois in merge_pairs.items():
                if r in sub_rois and merged_name not in used_merges:
                    new_roi_names.append(merged_name)
                    used_merges.add(merged_name)
                    replaced = True
                    break
                elif r in sub_rois:
                    replaced = True  # skip duplicate sub-ROI
                    break
            if not replaced:
                new_roi_names.append(r)
        roi_names = new_roi_names

    condition_colors = {'high': '#e67e22', 'low': '#3498db', 'contrast': '#2ecc71'}
    conditions = df['condition'].unique()
    eval_features = df['eval_feature'].unique()
    has_hemi = 'hemisphere' in df.columns and set(df['hemisphere'].unique()) != {'both'}

    bar_width = 0.25
    glm_tag = '_glm' if use_glm else ''

    if measure == 'perf':
        # One figure: ROIs as panels, eval features on x-axis, conditions as grouped bars
        n_panels = len(roi_names)
        if has_hemi:
            hemis = df['hemisphere'].unique()
            n_panels = len(roi_names) * len(hemis)

        fig, axes = plt.subplots(1, len(roi_names), figsize=(5 * len(roi_names), 5), squeeze=False)
        axes = axes[0]

        for roi_idx, roi_name in enumerate(roi_names):
            ax = axes[roi_idx]
            roi_df = df[df['roi'] == roi_name]

            if has_hemi:
                # Average across hemispheres for the bar plot
                roi_df = roi_df.groupby(['eval_feature', 'condition'], as_index=False)['mean_value'].mean()

            x = np.arange(len(eval_features))
            for c_idx, cond in enumerate(conditions):
                means = []
                sems = []
                for feat in eval_features:
                    subset = roi_df[(roi_df['eval_feature'] == feat) & (roi_df['condition'] == cond)]
                    if 'subject' in subset.columns and len(subset) > 1:
                        means.append(subset['mean_value'].mean())
                        sems.append(subset['mean_value'].sem())
                    else:
                        means.append(subset['mean_value'].mean() if len(subset) > 0 else 0)
                        sems.append(0)

                ax.bar(x + c_idx * bar_width, means, bar_width,
                       yerr=sems if any(s > 0 for s in sems) else None,
                       label=cond, color=condition_colors.get(cond, 'gray'), alpha=0.85,
                       capsize=3)

            ax.set_title(roi_name, fontsize=14, fontweight='normal')
            ax.set_xticks(x + bar_width * (len(conditions) - 1) / 2)
            ax.set_xticklabels(eval_features, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Mean $R^2$', fontsize=12)
            ax.legend(fontsize=10)
            ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        filepath = os.path.join(fig_dir, f'{sid}{file_label}_perf_roi_summary{glm_tag}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'saved: {filepath}')

    else:
        # One figure per model feature
        model_features = df['model_feature'].unique()
        for model_feat in model_features:
            mf_df = df[df['model_feature'] == model_feat]

            fig, axes = plt.subplots(1, len(roi_names), figsize=(5 * len(roi_names), 5), squeeze=False)
            axes = axes[0]

            for roi_idx, roi_name in enumerate(roi_names):
                ax = axes[roi_idx]
                roi_df = mf_df[mf_df['roi'] == roi_name]

                if has_hemi:
                    roi_df = roi_df.groupby(['eval_feature', 'condition'], as_index=False)['mean_value'].mean()

                x = np.arange(len(eval_features))
                for c_idx, cond in enumerate(conditions):
                    means = []
                    sems = []
                    for feat in eval_features:
                        subset = roi_df[(roi_df['eval_feature'] == feat) & (roi_df['condition'] == cond)]
                        if 'subject' in subset.columns and len(subset) > 1:
                            means.append(subset['mean_value'].mean())
                            sems.append(subset['mean_value'].sem())
                        else:
                            means.append(subset['mean_value'].mean() if len(subset) > 0 else 0)
                            sems.append(0)

                    ax.bar(x + c_idx * bar_width, means, bar_width,
                           yerr=sems if any(s > 0 for s in sems) else None,
                           label=cond, color=condition_colors.get(cond, 'gray'), alpha=0.85,
                           capsize=3)

                ax.set_title(f'{roi_name}\n{model_feat}', fontsize=12, fontweight='normal')
                ax.set_xticks(x + bar_width * (len(conditions) - 1) / 2)
                ax.set_xticklabels(eval_features, rotation=45, ha='right', fontsize=10)
                ax.set_ylabel('Mean product measure $R^2$', fontsize=12)
                ax.legend(fontsize=10)
                ax.axhline(y=0, color='black', linewidth=0.5)

            plt.tight_layout()
            filepath = os.path.join(
                fig_dir, f'{sid}{file_label}_ind_product_measure_roi_summary_{model_feat}{glm_tag}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'saved: {filepath}')
