#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ica100
#SBATCH --gres=gpu:3
#SBATCH --tasks-per-node=42
#SBATCH --time=8:00:00
#SBATCH -e sbatch_logs/error_%A_%a.log
#SBATCH -o sbatch_logs/output_%A_%a.log
#SBATCH --job-name=encoding
##### END OF JOB DEFINITION  #####
# Banded ridge voxelwise encoding model, one participant per array task (about 3 hours on an
# A100 per participant when saving weights). Also evaluates the fitted model on only the TRs
# with speaking (Supplementary Figure 8).
# Models used in the paper: joint_chunk120, joint_chunk120_simclr,
#                           vislang_gpt2_chunk120_eps0.3, vislang_simclr_gpt2_chunk120_eps0.3
# Usage (from the repository root), one array task per participant, at most 2 GPU jobs at once:
#   sbatch --array=1-34%2 scripts/sbatch/sbatch_encoding.sh joint_chunk120
source scripts/config.sh
MODEL=$1

subject=$(array_subject $PROJECT_DIR/analysis/participants.tsv)
subject=${subject#sub-}

activate_env
python -u -m src.encoding --s_num $subject \
	--dir $PROJECT_DIR --data_dir $DATA_DIR \
	--out_dir $PROJECT_DIR/analysis --figure_dir $PROJECT_DIR/figures \
	--model $MODEL \
	--smoothing-fwhm 3.0 \
	--mask ISC \
	--chunklen 120 \
	--save-weights 1 \
	--eval-mask-feature speaking \
	--zero-confounds-inner-cv
conda deactivate
