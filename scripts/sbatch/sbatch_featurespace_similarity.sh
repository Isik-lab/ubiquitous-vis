#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3:00:00
#SBATCH -e sbatch_logs/error_%A_%a.log
#SBATCH -o sbatch_logs/output_%A_%a.log
#SBATCH --job-name=featsim
##### END OF JOB DEFINITION  #####
# Cross-validated CCA similarity between two feature spaces (Figure 1C, Supplementary Figure 2),
# one pair per array task (rows of scripts/params/featurespace_comparisons.tsv).
# Usage (from the repository root), one array task per pair in scripts/params/featurespace_comparisons.tsv
# (1,081 pairs; the argument is the number of CCA latent dimensions, 1 in the paper):
#   sbatch --array=1-1081 scripts/sbatch/sbatch_featurespace_similarity.sh 1
source scripts/config.sh
LATENT=${1:-1}

features=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $PROJECT_DIR/scripts/params/featurespace_comparisons.tsv)

activate_env
python -u -m src.featurespace_similarity \
	--features $features \
	--dir $PROJECT_DIR \
	--out_dir $PROJECT_DIR/analysis --figure_dir $PROJECT_DIR/figures \
	--chunklen 120 \
	--method CCA \
	--latent_dim $LATENT
conda deactivate
