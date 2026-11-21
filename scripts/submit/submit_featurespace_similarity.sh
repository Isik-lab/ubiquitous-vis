#!/bin/bash
# Feature space similarity for every pair in scripts/params/featurespace_comparisons.tsv.
# Usage (from the repository root): bash scripts/submit/submit_featurespace_similarity.sh 1
#   (the argument is the number of CCA latent dimensions; the paper uses 1)
source scripts/config.sh
LATENT=${1:-1}
N=$(( $(wc -l < $PROJECT_DIR/scripts/params/featurespace_comparisons.tsv) - 1 ))
sbatch --array=1-$N scripts/sbatch/sbatch_featurespace_similarity.sh $LATENT
