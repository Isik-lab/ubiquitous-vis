#!/bin/bash
#SBATCH --nodes=1					# OpenMP requires a single node
#SBATCH --ntasks=1					# Run a single serial task
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=pc_interp_pooled			# Descriptive job name
##### END OF JOB DEFINITION  #####
# Correlates each pooled PC time series (movie embeddings projected onto the pooled PCs)
# with each annotated feature during speech, with a circular-shift permutation null.
# One array task per row of analysis/PCInterpretation/pooled_combinations.tsv, made from the pooled PCA combinations
# (subject column is the placeholder 'pooled'), after sbatch_pooled_pca.sh has finished:
#   awk -F'\t' -v OFS='\t' '{print "pooled", $3, $1, $2}' analysis/PooledPCA/pooled_pca_combinations.tsv \
#       > analysis/PCInterpretation/pooled_combinations.tsv
# then, once all array tasks are done: python -m src.pc_interpretation compile (results + FDR correction)
# Usage: sbatch --array=1-M%20 scripts/sbatch/sbatch_pc_interpretation_pooled.sh MODEL [POPULATION]

MODEL=${1:-vislang_gpt2_chunk120_eps0.3}
POPULATION=${2:-NT}
source scripts/config.sh

activate_env

# Parse this task's row (tab-separated: subject, feature_space, roi, roi_type)
COMBINATION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PROJECT_DIR/analysis/PCInterpretation/pooled_combinations.tsv)
IFS=$'\t' read -r SUBJECT FEATURE ROI ROI_TYPE <<< "$COMBINATION"

echo "Job running on node: $(hostname)" >> sbatch_logs/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
echo "Processing: $ROI ($ROI_TYPE) $FEATURE" >> sbatch_logs/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

python -u -m src.pc_interpretation run \
	--dir $PROJECT_DIR \
	--subject $SUBJECT \
	--feature_space $FEATURE \
	--roi $ROI \
	--roi_type $ROI_TYPE \
	--use_pooled_pca \
	--n_components 10 \
	--chunklen 120 \
	--method correlation \
	--fit_per_pc \
	--model $MODEL \
	--population $POPULATION \
	--use_speech_mask \
	--n_permutations 10000 \
	--block_size 120 \
	--perm_seed 42

conda deactivate
