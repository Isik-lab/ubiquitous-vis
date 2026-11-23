# Source data: main figures

Tabular source data are in the CSV files in this folder. Brain-map panels are plotted from the NIfTI files listed below. Paths are relative to the repository root. Files under `analysis/EncodingModel/`, `analysis/GLM/` and `analysis/SecondLevelIndividual/` are available on [Hugging Face](https://huggingface.co/datasets/hsmall/sherlock-and-localizers-fMRI).

All NIfTI files are in MNI152NLin2009cAsym space and were projected to the fsaverage surface for plotting.

## CSV files

| File | Figure panel |
|---|---|
| `Figure1C.csv` | 1C. Similarity between feature spaces (layer-averaged) |
| `Figure3B.csv` | 3B. Each participant's product measure of each feature in each ROI |
| `Figure4.csv` | 4. Each participant's response similarity for each plotted region pair and feature space |
| `Figure5.csv` | 5. Pooled weight PCA summary |

## Brain maps

### Figure 2. Group maps of encoding model performance (N = participant group average)

| Panel | Map | NIfTI |
|---|---|---|
| A | Cross-subject encoding model performance | `analysis/SecondLevelGroup/performance/sub-NT_brain2brain_encoding_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-perf_raw.nii.gz` |
| B | Joint encoding model performance | `analysis/SecondLevelGroup/performance/sub-NT_encoding_model-joint_chunk120_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-perf_raw.nii.gz` |
| C | Image (AlexNet) product measure | `analysis/SecondLevelGroup/ind_product_measure/sub-NT_encoding_model-joint_chunk120_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-ind_product_measure_raw.nii.gz` (AlexNet volume) |
| D | Multi-sentence (GPT-2) product measure | same file as C (GPT-2 volume) |

- **Panels A and B**: only voxels where the group mean is significantly above zero are shown. The significant voxels are given by the matching `..._measure-perf_p_fdr.nii.gz` file in the same folder (1 = significant; one-sided Wilcoxon signed-rank test across participants, FDR q < 0.001).
- **Panels C and D**: only voxels where the group mean is significantly above zero are shown.
  - The product measure file is 4D, with the feature on the first axis. Volume order: AlexNet, HuBERT, motion, GPT-2.
  - The significant voxels are given by `..._measure-product_measure_p_fdr.nii.gz` in the same folder (same volume order; 1 = significant; one-sided Wilcoxon signed-rank test across participants, FDR q < 0.001).

### Figure 3A. Preference maps

| Map | NIfTI |
|---|---|
| Group | `analysis/SecondLevelGroup/preference_map/sub-NT_encoding_model-joint_chunk120_smoothingfwhm-3.0_chunklen-120_mask-ISC_measure-preference1_map.nii.gz` |
| Example participant (sub-61) | computed from `analysis/EncodingModel/ind_product_measure/sub-61_encoding_model-joint_chunk120_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-ind_product_measure_raw.nii.gz` and `analysis/EncodingModel/performance/sub-61_encoding_model-joint_chunk120_smoothingfwhm-3.0_mask-ISC_chunklen-120_measure-perf_raw.nii.gz` |
| ROI outlines on the sub-61 map | `analysis/SecondLevelIndividual/localizer_masks/sub-61_smoothingfwhm-3.0_mask-pSTS_aSTS_glm_loc-interact-no_interact_run-all_binary.nii.gz` (social interaction, white) and `analysis/SecondLevelIndividual/localizer_masks/sub-61_smoothingfwhm-3.0_mask-pTemp_aTemp_frontal_glm_loc-intact-degraded_run-all_binary.nii.gz` (language, black) |

- **Group map values**: 0 means no preference; 1–4 are image (AlexNet), speech (HuBERT), motion and multi-sentence (GPT-2).
- **Participant map**:
  - Computed on the surface by `SecondLevelIndividual.generate_preference_maps` (`src/secondlevel_individual_encoding.py`).
  - Each feature space's product measure is the sum over its layers. The volume labels are in `analysis/EncodingModel/features/encoding_model-joint_chunk120_features.csv`.
  - Negative values are set to 0, and vertices are excluded where every feature is below 0.0001 or where performance is negative.
  - Each vertex is labelled with the feature that has the largest product measure.
