# Paths and environment for the cluster scripts. Edit these for your system.
# Every script in scripts/sbatch/ and scripts/submit/ sources this file, and expects to be
# run from the repository root, e.g.
#   bash scripts/submit/submit_encoding.sh joint_chunk120
# The #SBATCH partition/GPU lines at the top of each sbatch script are cluster specific too.

# this repository, with analysis/ and features/ downloaded from Hugging Face (https://huggingface.co/datasets/hsmall/sherlock-and-localizers-fMRI)
PROJECT_DIR=/path/to/ubiquitous-vis

# BIDS dataset (OpenNeuro ds006677), with fMRIPrep outputs in $DATA_DIR/derivatives
DATA_DIR=/path/to/ds006677

# fMRIPrep (only needed for scripts/sbatch/sbatch_fmriprep.sh)
FMRIPREP_IMG=/path/to/fmriprep-21.0.2.simg
FS_LICENSE=/path/to/freesurfer/license.txt

# conda environment created from environment.yml
CONDA_ENV=ubiquitous-vis

activate_env() {
    module --ignore_cache load anaconda 2>/dev/null || true
    conda activate "$CONDA_ENV"
}

# subject for this SLURM array task: line $SLURM_ARRAY_TASK_ID (after the header) of a participants file, e.g. sub-05
array_subject() {
    sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1" | cut -f1
}

mkdir -p sbatch_logs
