#!/bin/bash
#SBATCH --mail-type=END 			# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hsmall2@jhu.edu	# Your email address
#SBATCH --nodes=1					# OpenMP requires a single node
#SBATCH --ntasks=1					# Run a single serial task
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=fmriprep			# Descriptive job name
##### END OF JOB DEFINITION  #####
STUDY=$1 #first input is study folder

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /home/hsmall2/scratch4-lisik3/$STUDY/data/participants.tsv )

# Remove IsRunning files from FreeSurfer
find /home/hsmall2/scratch4-lisik3/$STUDY/data/derivatives/sourcedata/freesurfer/sub-$subject/ -name "*IsRunning*" -type f -delete


singularity run --cleanenv \
--bind /home/hsmall2/scratch4-lisik3/$STUDY/data:/data \
--bind /home/hsmall2/scratch4-lisik3/$STUDY/data/derivatives:/out \
/home/hsmall2/fmriprep-21.0.2.simg \
/data /out participant \
--participant_label $subject \
--n_cpus $SLURM_CPUS_PER_TASK \
--omp-nthreads 8 --nthreads 12 \
--fs-license-file /home/hsmall2/scratch4-lisik3/$STUDY/data/derivatives/freesurfer-6.0.1/license.txt \
--output-spaces MNI152NLin2009cAsym:res-2 \
--write-graph \
--cifti-output 170k \
--bids-database-dir /home/hsmall2 \
--use-syn-sdc \
-w /home/hsmall2/scratch4-lisik3/$STUDY/data/derivatives/tmp \

