#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=hsmall2@jhu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition shared
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH -e sbatch_logs/error_%A_%a.log
#SBATCH -o sbatch_logs/output_%A_%a.log
#SBATCH --job-name=cca_nochunk_control
##### CONTROL ANALYSIS: no-chunk (TR-level) nested CV.                    #####
##### Mirrors sbatch_featurespace_similarity_pipeline.sh so that pair     #####
##### selection is compatible with the existing pipeline.                 #####
LATENT=$1      # first arg: latent dim (match Fig 1C default = 1)
PAIRS_TSV=${2:-/home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/scripts/featurespace_comparisons_fig1c_pairs.tsv}

module --ignore_cache load "anaconda"
conda activate naturalistic-multimodal-movie-pip

features=$( sed -n "$((${SLURM_ARRAY_TASK_ID} + 1))p" "$PAIRS_TSV")

python -u -m src.cca_similarity_cv_nochunk_control \
    --mode pair \
    --features "$features" \
    --dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie \
    --out_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/analysis \
    --figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/figures \
    --chunklen 120 \
    --latent_dim "$LATENT" \
    --outer-folds 5 \
    --inner-folds 5

conda deactivate
