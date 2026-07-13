#!/bin/bash
# Wrapper: submit the NON-CV in-sample CCA control for the joint_chunk120
# encoding model with automatic array sizing.
# Usage: bash scripts/submit_cca_noncv_control_joint_chunk120.sh
#
# Each task runs one feature pair from the TSV; the sbatch script reads
# line ${SLURM_ARRAY_TASK_ID}+1, so the array is 0-based.
# %50 caps concurrency at 50 concurrent tasks to share cluster resources.

PAIRS_TSV="/home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/scripts/featurespace_comparisons_joint_chunk120.tsv"
LATENT_DIM=1

if [ ! -f "$PAIRS_TSV" ]; then
    echo "ERROR: pair TSV not found: $PAIRS_TSV"
    echo "Generate it with: python -m scripts.generate_fig1c_pair_tsv --model joint_chunk120 --output $PAIRS_TSV"
    exit 1
fi

N_PAIRS=$( wc -l "$PAIRS_TSV" | cut -f1 -d' ' )
LAST_IDX=$(( N_PAIRS - 1 ))

echo "Found $N_PAIRS pairs in $PAIRS_TSV"
echo "Submitting non-CV control array 0-$LAST_IDX (throttle %50) for joint_chunk120"

sbatch --array=0-${LAST_IDX}%50 sbatch/sbatch_cca_similarity_noncv_control.sh $LATENT_DIM $PAIRS_TSV

echo "Job submitted."
