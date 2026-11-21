#!/bin/bash
# Preprocess every participant in analysis/participants.tsv with fMRIPrep.
# Usage (from the repository root): bash scripts/submit/submit_fmriprep.sh
source scripts/config.sh
N=$(( $(wc -l < $PROJECT_DIR/analysis/participants.tsv) - 1 ))
sbatch --array=1-$N scripts/sbatch/sbatch_fmriprep.sh
