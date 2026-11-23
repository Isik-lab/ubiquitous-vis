#!/bin/bash
#SBATCH --job-name=resp_sim
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=shared
#SBATCH --output=sbatch_logs/resp_sim_%j_out.log
#SBATCH --error=sbatch_logs/resp_sim_%j_err.log

# Response Similarity Analysis (Main Figure 4) - Cluster Submission Script
#
# Computes the within-subject correlation of encoding model weights
# between functionally localized brain regions.
#
# Usage (from the project root); Main Figure 4 is one job per feature space, with these AXES and FILEPATH_TAG:
#   "motion:motion" motion
#   "alexnet:alexnet_eps0.3_layer5,alexnet_eps0.3_layer6,alexnet_eps0.3_layer7" alexnet_eps0.3
#   "hubert:hubert_chunk120_layer10,hubert_chunk120_layer11,hubert_chunk120_layer12" hubert_chunk120
#   "GPT2:GPT2_chunk120_layer10,GPT2_chunk120_layer11,GPT2_chunk120_layer12" GPT2_chunk120
# e.g.
#   sbatch scripts/sbatch/sbatch_response_similarity.sh vislang_gpt2_chunk120_eps0.3 weights NT "motion:motion" motion

set -e

MODEL=${1:-vislang_gpt2_chunk120_eps0.3}
RESPONSE=${2:-weights}
POPULATION=${3:-NT}
AXES=${4:-"motion:motion"}
FILEPATH_TAG=${5:-""}

source scripts/config.sh
activate_env

echo "=========================================="
echo "Response Similarity Analysis"
echo "  Model: $MODEL"
echo "  Response: $RESPONSE"
echo "  Population: $POPULATION"
echo "  Axes: $AXES"
echo "  Filepath tag: $FILEPATH_TAG"
echo "=========================================="

python -u -m src.response_similarity \
    --dir $PROJECT_DIR \
    --out_dir $PROJECT_DIR/analysis \
    --figure_dir $PROJECT_DIR/figures/subfigures \
    --model $MODEL \
    --response $RESPONSE \
    --population $POPULATION \
    --axes $AXES \
    --filepath_tag "$FILEPATH_TAG" \
    --average_posterior_anterior \
    --no_split_hemi \
    --smoothing_fwhm 3.0 \
    --chunklen 120 \
    --mask ISC \
    --perc_top_voxels 5 \
    --space MNI152NLin2009cAsym

echo "Completed: $AXES"
