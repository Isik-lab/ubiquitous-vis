#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=12
#SBATCH --time=3:30:00
#SBATCH -e sbatch_logs/error_%A_%a.log
#SBATCH -o sbatch_logs/output_%A_%a.log
#SBATCH --job-name=b2b
##### END OF JOB DEFINITION  #####
# Cross-subject encoding: predict each held-out participant's responses from the other
# participants' SRP time series (Figure 2A), one participant per array task.
# Needs the srp time series: bash scripts/submit/submit_timeseries.sh srp ISC
# Usage: bash scripts/submit/submit_brain2brain.sh ISC
source scripts/config.sh
MASK=${1:-ISC}

subject=$(array_subject $PROJECT_DIR/analysis/participants.tsv)

activate_env
python -u -m src.brain2brain --s_num $subject \
	--dir $PROJECT_DIR --data_dir $DATA_DIR \
	--out_dir $PROJECT_DIR/analysis --figure_dir $PROJECT_DIR/figures \
	--smoothing-fwhm 3.0 \
	--population NT \
	--mask $MASK \
	--type encoding \
	--chunklen 120 \
	--zero-confounds-inner-cv
conda deactivate
