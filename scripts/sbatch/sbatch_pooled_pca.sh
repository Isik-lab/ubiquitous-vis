#!/bin/bash
#SBATCH --nodes=1					# OpenMP requires a single node
#SBATCH --ntasks=1					# Run a single serial task
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=pooled_pca			# Descriptive job name
##### END OF JOB DEFINITION  #####
# Pooled PCA of encoding model weights: one array task per ROI x feature_space row of
# analysis/PooledPCA/pooled_pca_combinations.tsv (see scripts/submit/submit_pooled_pca_workflow.sh).
# feature_space may join several layers with '+', which are concatenated into one PCA.
# Usage: sbatch --array=1-N scripts/sbatch/sbatch_pooled_pca.sh MODEL [POPULATION]

MODEL=${1:-vislang_gpt2_chunk120_eps0.3}
POPULATION=${2:-NT}
source scripts/config.sh

activate_env

# Parse this task's row (tab-separated: roi, roi_type, feature_space)
COMBINATION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PROJECT_DIR/analysis/PooledPCA/pooled_pca_combinations.tsv)
IFS=$'\t' read -r ROI ROI_TYPE FEATURE <<< "$COMBINATION"

echo "Job running on node: $(hostname)" >> sbatch_logs/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
echo "Processing: $ROI ($ROI_TYPE) $FEATURE" >> sbatch_logs/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

python -u -m src.pooled_pca run \
	--dir $PROJECT_DIR \
	--feature_spaces ${FEATURE//+/ } \
	--roi $ROI \
	--roi_type $ROI_TYPE \
	--population $POPULATION \
	--n_components 0.95 \
	--min_performance 0.0 \
	--model $MODEL \
	--smoothing_fwhm 3.0 \
	--chunklen 120 \
	--mask ISC

conda deactivate
