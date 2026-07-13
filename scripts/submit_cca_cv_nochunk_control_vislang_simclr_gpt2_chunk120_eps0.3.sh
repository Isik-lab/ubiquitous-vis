#!/bin/bash
# Wrapper: submit the NO-CHUNK (TR-level) nested-CV CCA control for the
# vislang_simclr_gpt2_chunk120_eps0.3 encoding model with automatic array
# sizing.
# Usage: bash scripts/submit_cca_cv_nochunk_control_vislang_simclr_gpt2_chunk120_eps0.3.sh
#
# Each task runs one feature pair through full 5x5 nested CV with the 25-cell
# reg grid (~11 min per pair). The sbatch script reads line
# ${SLURM_ARRAY_TASK_ID}+1 so the array is 0-based.
# %30 caps concurrency to be a good cluster citizen.

PAIRS_TSV="/home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/scripts/featurespace_comparisons_vislang_simclr_gpt2_chunk120_eps0.3.tsv"
LATENT_DIM=1

if [ ! -f "$PAIRS_TSV" ]; then
    echo "ERROR: pair TSV not found: $PAIRS_TSV"
    echo "Generate it with: python -m scripts.generate_fig1c_pair_tsv --model vislang_simclr_gpt2_chunk120_eps0.3 --output $PAIRS_TSV"
    exit 1
fi

N_PAIRS=$( wc -l "$PAIRS_TSV" | cut -f1 -d' ' )
LAST_IDX=$(( N_PAIRS - 1 ))

echo "Found $N_PAIRS pairs in $PAIRS_TSV"
echo "Submitting no-chunk CV control array 0-$LAST_IDX (throttle %30) for vislang_simclr_gpt2_chunk120_eps0.3"

sbatch --array=0-${LAST_IDX}%30 sbatch/sbatch_cca_similarity_cv_nochunk_control.sh $LATENT_DIM $PAIRS_TSV

echo "Job submitted."
