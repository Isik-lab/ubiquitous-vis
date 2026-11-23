#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=48G
#SBATCH --time=10:00
#SBATCH -e sbatch_logs/error_%A_%a.log
#SBATCH -o sbatch_logs/output_%A_%a.log
#SBATCH --job-name=timeseries
##### END OF JOB DEFINITION  #####
# Extract each participant's movie time series, one participant per array task.
#   isc: whole brain, 6mm smoothing, denoised and z-scored within each run (used for the leave-one-out ISC mask)
#   srp: 3mm smoothing, sparse random projection within MASK, not denoised or z-scored (input to cross-subject encoding)
# Usage: bash scripts/submit/submit_timeseries.sh isc
#        bash scripts/submit/submit_timeseries.sh srp ISC
source scripts/config.sh
MODE=$1
MASK=$2

subject=$(array_subject $PROJECT_DIR/analysis/participants.tsv)

activate_env
if [ "$MODE" == "isc" ]; then
	python -u -m src.timeseries --s_num $subject \
		--dir $PROJECT_DIR --data_dir $DATA_DIR \
		--out_dir $PROJECT_DIR/analysis --figure_dir $PROJECT_DIR/figures \
		--smoothing-fwhm 6.0 \
		--denoise 1 \
		--norm_runs 1 \
		--srp 0
else
	python -u -m src.timeseries --s_num $subject \
		--dir $PROJECT_DIR --data_dir $DATA_DIR \
		--out_dir $PROJECT_DIR/analysis --figure_dir $PROJECT_DIR/figures \
		--smoothing-fwhm 3.0 \
		--mask $MASK \
		--srp 1
fi
conda deactivate
