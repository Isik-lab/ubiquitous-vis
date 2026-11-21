#!/bin/bash
# Extract the movie time series of every participant.
# Usage (from the repository root):
#   bash scripts/submit/submit_timeseries.sh isc       # whole brain, for the ISC mask
#   bash scripts/submit/submit_timeseries.sh srp ISC   # SRP within the ISC mask, for cross-subject encoding
source scripts/config.sh
N=$(( $(wc -l < $PROJECT_DIR/analysis/participants.tsv) - 1 ))
sbatch --array=1-$N scripts/sbatch/sbatch_timeseries.sh "$@"
