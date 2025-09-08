#!/bin/bash
#SBATCH --mail-type=END 			# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hsmall2@jhu.edu	# Your email address
#SBATCH -A lisik33
#SBATCH --nodes=1					# OpenMP requires a single node
#SBATCH --ntasks=1					# Run a single serial task
#SBATCH --mem-per-cpu=48G
#SBATCH --time=5:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=srp_timeseries			# Descriptive job name
##### END OF JOB DEFINITION  #####
STUDY=$1 #first input is study folder
MASK=$2 #second input is the mask we are doing the SRP in

module --ignore_cache load "anaconda"
# ml anaconda
conda activate naturalistic-multimodal-movie-pip

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))p" /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/analysis/participants_b2b.tsv )

echo "Job running on node: $(hostname)" >> sbatch_logs/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

python -u submit_timeseries.py --s_num $subject \
		--dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie \
		--data_dir /home/hsmall2/scratch4-lisik3/$STUDY/data \
		--out_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/analysis \
		--figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/figures \
		--smoothing-fwhm 3.0 \
        --mask $MASK \
		--srp 1

conda deactivate