#!/bin/bash
# Fit an encoding model for every participant (at most 2 GPU jobs at once).
# Usage (from the repository root): bash scripts/submit/submit_encoding.sh joint_chunk120
source scripts/config.sh
MODEL=$1
N=$(( $(wc -l < $PROJECT_DIR/analysis/participants.tsv) - 1 ))
sbatch --array=1-${N}%2 scripts/sbatch/sbatch_encoding.sh $MODEL
