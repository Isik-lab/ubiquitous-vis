#!/bin/bash
# Cross-subject encoding for every participant (at most 6 GPU jobs at once).
# Usage (from the repository root): bash scripts/submit/submit_brain2brain.sh ISC
source scripts/config.sh
MASK=${1:-ISC}
N=$(( $(wc -l < $PROJECT_DIR/analysis/participants.tsv) - 1 ))
sbatch --array=1-${N}%6 scripts/sbatch/sbatch_brain2brain.sh $MASK
