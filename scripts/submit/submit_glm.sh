#!/bin/bash
# Fit the first-level localizer GLMs of every participant who did the task.
# Usage (from the repository root):
#   bash scripts/submit/submit_glm.sh SIpointlights
#   bash scripts/submit/submit_glm.sh language
source scripts/config.sh
TASK=$1
if [ "$TASK" == "language" ]; then
	PARTICIPANTS=$PROJECT_DIR/analysis/participants_language.tsv
else
	PARTICIPANTS=$PROJECT_DIR/analysis/participants.tsv
fi
N=$(( $(wc -l < $PARTICIPANTS) - 1 ))
sbatch --array=1-$N scripts/sbatch/sbatch_glm.sh $TASK
