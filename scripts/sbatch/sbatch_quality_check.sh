#!/bin/bash
#SBATCH --mail-type=END 			# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hsmall2@jhu.edu	# Your email address
#SBATCH --nodes=1					# OpenMP requires a single node
#SBATCH --ntasks=1					# Run a single serial task
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=qualcheck		# Descriptive job name
##### END OF JOB DEFINITION  #####
POP=$1 #first input is the population

ml anaconda
conda activate naturalistic-multimodal-movie-pip

python -u submit_quality_check.py \
		--dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie \
		--data_dir /home/hsmall2/scratch4-lisik3/Sherlock_ASD/data \
		--out_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/analysis \
		--figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/figures \
		--task sherlock \
		--population $POP

conda deactivate