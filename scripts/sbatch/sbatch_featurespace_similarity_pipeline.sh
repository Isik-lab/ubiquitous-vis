#!/bin/bash
#SBATCH --mail-type=END 			# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hsmall2@jhu.edu	# Your email address
#SBATCH --nodes=1					# OpenMP requires a single node
#SBATCH --ntasks-per-node=1
#SBATCH --partition shared
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3:00:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=feat_reg			# Descriptive job name
##### END OF JOB DEFINITION  #####
LATENT=$1 #first input is the number of latent dimensions to use for CCA

module --ignore_cache load "anaconda"
# ml anaconda
conda activate naturalistic-multimodal-movie-pip

features=$( sed -n "$((${SLURM_ARRAY_TASK_ID} + 1))p" /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/scripts/featurespace_comparisons.tsv)

python -u submit_featurespace_similarity.py \
		--features $features \
		--dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie \
		--out_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/analysis \
		--figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/figures \
		--chunklen 20 \
		--method CCA \
		--latent_dim $LATENT

conda deactivate