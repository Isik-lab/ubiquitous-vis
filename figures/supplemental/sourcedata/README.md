# Source data: supplementary figures

Tabular source data are in the CSV files in this folder. Brain-map panels are plotted from the NIfTI files listed below. Paths are relative to the repository root. Files under `analysis/EncodingModel/`, `analysis/GLM/` and `analysis/SecondLevelIndividual/` are available on [Hugging Face](https://huggingface.co/datasets/hsmall/sherlock-and-localizers-fMRI).

All NIfTI files are in MNI152NLin2009cAsym space and were projected to the fsaverage surface for plotting.

## CSV files

| File | Figure panel |
|---|---|
| `SupplementaryFigure2.csv` | 2. Similarity between feature spaces (layerwise) |
| `SupplementaryFigure5.csv` | 5. Each participant's localizer responses in each ROI |
| `SupplementaryFigure6A.csv` | 6A. Cross-subject encoding performance in each ROI (outliers more than 3×IQR from the quartiles excluded) |
| `SupplementaryFigure6B.csv` | 6B. Vision (AlexNet) and language product measures in each ROI |
| `SupplementaryFigure6C.csv` | 6C. Vision (SimCLR) and language product measures in each ROI |
| `SupplementaryFigure7.csv` | 7. Voxel-wise image (AlexNet) and sentence (GPT-2) product measures in each ROI |
| `SupplementaryFigure8A.csv` | 8A. Product measures of each feature space in each ROI, AlexNet joint model |
| `SupplementaryFigure8B.csv` | 8B. The same for the SimCLR joint model |
| `SupplementaryFigure9.csv` | 9. Product measures of the simplified joint model (motion, AlexNet, HuBERT, word2vec, GPT-2) in the full movie vs. speech-only timepoints |
| `SupplementaryFigure10.csv` | 10. Cumulative variance explained by the PCs of each feature space |
| `SupplementaryFigure11.csv` | 11. Pooled weight PCA, per PC |

## Brain maps

### Supplementary Figure 1. Localizer contrasts: example participant (sub-33) and group

| Contrast | sub-33 | Group |
|---|---|---|
| Motion (interacting + not interacting point-light figures) | `analysis/GLM/sub-33/sub-33_task-SIpointlights_space-MNI152NLin2009cAsym_run-123_measure-zscore_contrast-interact&no_interact.nii.gz` | `analysis/SecondLevelGroup/glm_zscores/sub-NT_smoothingfwhm-3.0_measure-zscore_contrast-interact&no_interact.nii.gz` |
| Social interaction (interacting − not interacting) | `analysis/GLM/sub-33/sub-33_task-SIpointlights_space-MNI152NLin2009cAsym_run-123_measure-zscore_contrast-interact-no_interact.nii.gz` | `analysis/SecondLevelGroup/glm_zscores/sub-NT_smoothingfwhm-3.0_measure-zscore_contrast-interact-no_interact.nii.gz` |
| Language (intact − degraded speech) | `analysis/GLM/sub-33/sub-33_task-language_space-MNI152NLin2009cAsym_run-12_measure-zscore_contrast-intact-degraded.nii.gz` | `analysis/SecondLevelGroup/glm_zscores/sub-NT_smoothingfwhm-3.0_measure-zscore_contrast-intact-degraded.nii.gz` |

All maps are z-scores. The sub-33 maps are thresholded at z > 3, and the group maps at FDR q < 0.01 (one-sided).

### Supplementary Figure 3. Product measure of each feature space: example participant (sub-33) and group

| Map | NIfTI |
|---|---|
| sub-33 | `analysis/EncodingModel/ind_product_measure/sub-33_encoding_model-joint_chunk120_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-ind_product_measure_raw.nii.gz` |
| Group | `analysis/SecondLevelGroup/ind_product_measure/sub-NT_encoding_model-joint_chunk120_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-ind_product_measure_raw.nii.gz` |

- **sub-33 file**:
  - One volume per layer or feature. The volume labels are in `analysis/EncodingModel/features/encoding_model-joint_chunk120_features.csv`.
  - Each feature space's map is the sum over its layers. It is thresholded at 0.01.
- **Group file**:
  - Volume order: AlexNet, HuBERT, motion, GPT-2.
  - Only voxels that pass FDR q < 0.001 are shown (one-sided Wilcoxon signed-rank test across participants). The significant voxels are given by `..._measure-product_measure_p_fdr.nii.gz` in the same folder (same volume order; 1 = significant). No HuBERT voxel passes.
  - These are the same group maps as in Main Figure 2C–D.

### Supplementary Figure 4. Every participant's preference map

Each participant's map is computed in the same way as the participant map in Main Figure 3A (see `figures/main/sourcedata/README.md`), from these files:
- `analysis/EncodingModel/ind_product_measure/<participant>_encoding_model-joint_chunk120_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-ind_product_measure_raw.nii.gz`
- `analysis/EncodingModel/performance/<participant>_encoding_model-joint_chunk120_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-perf_raw.nii.gz`

The ROI outlines are the participant's files in `analysis/SecondLevelIndividual/localizer_masks/`.
