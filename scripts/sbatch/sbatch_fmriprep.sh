#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH -e sbatch_logs/error_%A_%a.log
#SBATCH -o sbatch_logs/output_%A_%a.log
#SBATCH --job-name=fmriprep
##### END OF JOB DEFINITION  #####
# fMRIPrep 21.0.2 preprocessing, one participant per array task (rows of analysis/participants.tsv).
# Usage: bash scripts/submit/submit_fmriprep.sh
source scripts/config.sh

subject=$(array_subject $PROJECT_DIR/analysis/participants.tsv)
subject=${subject#sub-}

# remove stale FreeSurfer lock files from a previous run
find $DATA_DIR/derivatives/sourcedata/freesurfer/sub-$subject/ -name "*IsRunning*" -type f -delete 2>/dev/null

singularity run --cleanenv \
	--bind $DATA_DIR:/data \
	--bind $DATA_DIR/derivatives:/out \
	$FMRIPREP_IMG \
	/data /out participant \
	--participant_label $subject \
	--n_cpus $SLURM_CPUS_PER_TASK \
	--omp-nthreads 8 --nthreads 12 \
	--fs-license-file $FS_LICENSE \
	--output-spaces MNI152NLin2009cAsym:res-2 \
	--write-graph \
	--cifti-output 170k \
	--bids-database-dir $DATA_DIR/derivatives/tmp/bids_db \
	--use-syn-sdc \
	-w $DATA_DIR/derivatives/tmp
