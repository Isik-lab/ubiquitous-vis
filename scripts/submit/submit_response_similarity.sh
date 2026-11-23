#!/bin/bash
# Submit the response similarity analyses of Main Figure 4 to the cluster:
# one job per feature space, correlating the weights of the last three layers of each DNN
# (and the motion energy features) between the localized regions within each participant.
#
# Usage (from the project root):
#   bash scripts/submit/submit_response_similarity.sh
#   bash scripts/submit/submit_response_similarity.sh --population NT --model vislang_gpt2_chunk120_eps0.3 --response weights

set -e

POPULATION="NT"
MODEL="vislang_gpt2_chunk120_eps0.3"
RESPONSE="weights"

while [[ $# -gt 0 ]]; do
    case $1 in
        --population) POPULATION="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --response) RESPONSE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

source scripts/config.sh

echo "Submitting response similarity analyses"
echo "  Population: $POPULATION"
echo "  Model: $MODEL"
echo "  Response: $RESPONSE"
echo ""

# "<feature space>:<features whose weights are correlated>" "<filepath_tag>"
JOB1=$(sbatch --parsable scripts/sbatch/sbatch_response_similarity.sh \
    "$MODEL" "$RESPONSE" "$POPULATION" \
    "motion:motion" "motion")
echo "  Submitted motion: $JOB1"

JOB2=$(sbatch --parsable scripts/sbatch/sbatch_response_similarity.sh \
    "$MODEL" "$RESPONSE" "$POPULATION" \
    "alexnet:alexnet_eps0.3_layer5,alexnet_eps0.3_layer6,alexnet_eps0.3_layer7" "alexnet_eps0.3")
echo "  Submitted AlexNet layers 5,6,7: $JOB2"

JOB3=$(sbatch --parsable scripts/sbatch/sbatch_response_similarity.sh \
    "$MODEL" "$RESPONSE" "$POPULATION" \
    "hubert:hubert_chunk120_layer10,hubert_chunk120_layer11,hubert_chunk120_layer12" "hubert_chunk120")
echo "  Submitted HuBERT layers 10,11,12: $JOB3"

JOB4=$(sbatch --parsable scripts/sbatch/sbatch_response_similarity.sh \
    "$MODEL" "$RESPONSE" "$POPULATION" \
    "GPT2:GPT2_chunk120_layer10,GPT2_chunk120_layer11,GPT2_chunk120_layer12" "GPT2_chunk120")
echo "  Submitted GPT-2 layers 10,11,12: $JOB4"

echo ""
echo "Monitor with: squeue -u \$USER | grep resp_sim"
echo "Logs in: sbatch_logs/"
