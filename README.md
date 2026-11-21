# ubiquitous-vis
Code for the paper 'Ubiquitous cortical sensitivity to visual information during naturalistic, audiovisual movie viewing'. [Read the paper here!](https://osf.io/preprints/psyarxiv/b5p4n_v1)

## Setup
```
conda env create -f environment.yml
conda activate ubiquitous-vis
pip install -e .        # from the repository root; makes `src` importable
```

## Data
- **Raw fMRI** (BIDS) will be [available on OpenNeuro](https://openneuro.org/datasets/ds006677).
- **Movie features and analysis outputs** (first-level encoding models and GLMs, second-level results) are [on Hugging Face](https://huggingface.co/datasets/hsmall/sherlock-and-localizers-fMRI). Download them into the repository root (`huggingface-cli download hsmall/sherlock-and-localizers-fMRI --repo-type dataset --local-dir .`) to rerun the second-level analyses. The dataset card lists what is included.
- **In this repository** are only the files needed to make the paper figures: the subfigures and summary files that `Plot Figures.ipynb` assembles, the ROI parcels, the participant lists, and the final figures (with source data) in `figures/main` and `figures/supplemental`.

## Repository layout
```
src/                         all analysis code (importable as `src`, runnable as `python -m src.<module>`)
scripts/
  Plot Figures.ipynb         makes every main and supplementary figure (runs on a fresh clone)
  SecondLevel Analyses.ipynb group and ROI analyses that produce the subfigures (needs the Hugging Face data)
  config.sh                  paths and conda environment for the cluster scripts (edit this first)
  sbatch/                    one SLURM job script per first-level step
  submit/                    submits each job script as an array over participants or feature pairs
  params/                    GLM contrasts and the feature-space pairs for the similarity analysis
analysis/                    participant lists and ROI parcels (+ Hugging Face outputs once downloaded)
figures/                     subfigures and final figures
tables/                      supplementary tables (LaTeX)
```

## Running the analyses
### 1. First level (cluster)
Edit `scripts/config.sh` (`PROJECT_DIR`, `DATA_DIR`, conda environment, fMRIPrep image) and the `#SBATCH` partition lines for your cluster, then run each step from the repository root. Each submit script launches one job per participant (or per feature pair).

| step | command | module |
|---|---|---|
| preprocessing (fMRIPrep 21.0.2) | `bash scripts/submit/submit_fmriprep.sh` | – |
| time series for the ISC mask | `bash scripts/submit/submit_timeseries.sh isc` | `src/timeseries.py` |
| SRP time series for cross-subject encoding | `bash scripts/submit/submit_timeseries.sh srp ISC` | `src/timeseries.py` |
| localizer GLMs | `bash scripts/submit/submit_glm.sh SIpointlights`<br>`bash scripts/submit/submit_glm.sh language` | `src/glm.py` |
| encoding models (about 3 h per participant on an A100 when saving weights) | `bash scripts/submit/submit_encoding.sh joint_chunk120`<br>(also `joint_chunk120_simclr`, `vislang_gpt2_chunk120_eps0.3`, `vislang_simclr_gpt2_chunk120_eps0.3`) | `src/encoding.py` |
| cross-subject encoding | `bash scripts/submit/submit_brain2brain.sh ISC` | `src/brain2brain.py` |
| feature space similarity | `bash scripts/submit/submit_featurespace_similarity.sh 1` | `src/featurespace_similarity.py` |
| encoding weight similarity across regions | `bash scripts/submit/submit_response_similarity.sh` | `src/response_similarity.py` |
| pooled weight PCA and its interpretation | `bash scripts/submit/submit_pooled_pca_workflow.sh` | `src/pooled_pca.py`, `src/pc_interpretation.py` |

The model names (which feature spaces make up each encoding model) are defined in `src/helpers.py:get_models_dict`.

### 2. Second level
`scripts/SecondLevel Analyses.ipynb` runs the group and ROI analyses on the first-level outputs and saves the subfigures to `figures/subfigures`. Each section notes which data folders it needs.

### 3. Figures
`scripts/Plot Figures.ipynb` assembles every figure except Figure 1 into `figures/main` and `figures/supplemental` and writes the source data for each figure to their `sourcedata/` folders. It runs on a fresh clone of this repository. Figure 1 (`figures/main/fig1_final_manuscript.pdf`) was assembled by hand; its panel B is `figures/main/featurespace_correlation_layers_averaged.pdf`, made by Plot Figures.

| figure | analysis (notebook section) | main modules |
|---|---|---|
| 1B, S2 feature space similarity | cluster step above | `featurespace_similarity` |
| 2 cross-subject and joint encoding model maps, S3 | group maps of encoding performance | `brain2brain`, `secondlevel_group_encoding` |
| 3 variance decomposition in ROIs, S6, S8 | joint model variance decomposition | `secondlevel_individual_encoding` |
| 4 encoding weight similarity across regions | feature tuning similarity across regions | `response_similarity`, `secondlevel_individual_encoding` |
| 5 interpreting the pooled weight PCs, S11 | pooled weight PCA | `pooled_pca`, `pc_interpretation`, `weight_pca_analysis` |
| S1 localizer maps, S5 localizer responses in ROIs | brain maps of controlled stimuli tasks | `glm`, `secondlevel_group_glm` |
| S4 preference maps of every participant | every participant's preference map | `secondlevel_individual_encoding` |
| S7 voxelwise image vs. sentence product measures | supplementary ROI analyses | `secondlevel_individual_encoding` |
| S9 full movie vs. speaking-only evaluation (simplified joint model) | speech-only evaluation | `encoding` (`--eval-mask-feature speaking`), `feature_mask_eval` |
| S10 feature space dimensionality | dimensionality of the feature spaces | `intrinsic_dimensionality` |

### 4. Supplementary tables
`scripts/SecondLevel Analyses.ipynb` writes the statistics tables as LaTeX to `tables/` (`SecondLevelIndividual.glm_voxel_selection` and `compute_response_similarity` in `src/secondlevel_individual_encoding.py`).

| table | file | notebook section |
|---|---|---|
| S1 feature space dimensionality | written by hand | – |
| S2 pairwise comparisons of region response profiles | `pairwise_anova_joint_chunk120__all.tex` | joint model variance decomposition (Figure 3B) |
| S3 feature predictivity in each region compared to zero | `lme_compare_to_zero_joint_chunk120__all.tex` | same |
| S4 motion weight similarity across regions | `response_similarity_pairwise_comparisons_motion_hemis_together.tex` | feature tuning similarity across regions (Figure 4) |
| S5 AlexNet (layers 5–7) weight similarity | `response_similarity_pairwise_comparisons_alexnet_hemis_together.tex` | same |
| S6 HuBERT (layers 10–12) weight similarity | `response_similarity_pairwise_comparisons_hubert_hemis_together.tex` | same |
| S7 GPT-2 (layers 10–12) weight similarity | `response_similarity_pairwise_comparisons_GPT2_hemis_together.tex` | same |

The other files in `tables/` are the same comparisons for the supplementary ROI figures, which the paper doesn't print as tables: `pairwise_anova_joint_chunk120__supp.tex` (S5/S7 section), `pairwise_anova_joint_chunk120{,_simclr}_vision_vs_language_.tex` and `lme_compare_features_joint_chunk120{,_simclr}_vision_vs_language_.tex` (S6B, C), and `pairwise_anova_joint_chunk120{,_simclr}__sep_post_ant.tex` and `lme_compare_to_zero_joint_chunk120{,_simclr}__sep_post_ant.tex` (S8).

## Tracked data
`analysis/`, `features/`, `figures/` and `tables/` are listed in `.gitignore`, because the full outputs live on Hugging Face. The files from these folders that are in git were added with `git add -f`. To track another one, add it the same way.
