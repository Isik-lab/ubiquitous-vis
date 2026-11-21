#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00:30:00
#SBATCH -e sbatch_logs/error_%A_%a.log
#SBATCH -o sbatch_logs/output_%A_%a.log
#SBATCH --job-name=glm
##### END OF JOB DEFINITION  #####
# First-level GLM of a localizer task, one participant per array task.
#   SIpointlights: social interaction localizer (rows of analysis/participants.tsv)
#   language:      language localizer (rows of analysis/participants_language.tsv)
# Contrasts and run groupings are in scripts/params/glm_<task>.json.
# Usage: bash scripts/submit/submit_glm.sh SIpointlights
source scripts/config.sh
TASK=$1

if [ "$TASK" == "language" ]; then
	subject=$(array_subject $PROJECT_DIR/analysis/participants_language.tsv)
else
	subject=$(array_subject $PROJECT_DIR/analysis/participants.tsv)
fi
subject=${subject#sub-}

activate_env
python -u -m src.glm --s_num $subject \
	--dir $PROJECT_DIR --data_dir $DATA_DIR \
	--out_dir $PROJECT_DIR/analysis --figure_dir $PROJECT_DIR/figures \
	--smoothing-fwhm 3.0 \
	--mask None \
	--task $TASK
conda deactivate
