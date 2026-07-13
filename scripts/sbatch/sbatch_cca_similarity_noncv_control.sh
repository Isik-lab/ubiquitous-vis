#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=hsmall2@jhu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition shared
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00
#SBATCH -e sbatch_logs/error_%A_%a.log
#SBATCH -o sbatch_logs/output_%A_%a.log
#SBATCH --job-name=cca_noncv_control
##### CONTROL ANALYSIS: non-CV (in-sample) CCA per pair.                  #####
##### Runs one feature pair per SLURM array task, writing a small CSV     #####
##### that a later --mode compile-model step aggregates.                  #####
LATENT=$1        # first arg: latent dim (match Fig 1C default = 1)
PAIRS_TSV=$2     # second arg: TSV listing one "f1-f2" per line

module --ignore_cache load "anaconda"
conda activate naturalistic-multimodal-movie-pip

features=$( sed -n "$((${SLURM_ARRAY_TASK_ID} + 1))p" "$PAIRS_TSV")

python -u -m src.cca_similarity_noncv_control \
    --mode pair \
    --features "$features" \
    --dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie \
    --out_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/analysis \
    --figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/figures \
    --chunklen 120 \
    --latent_dim "$LATENT" \
    --reg-param 1e-5

conda deactivate
