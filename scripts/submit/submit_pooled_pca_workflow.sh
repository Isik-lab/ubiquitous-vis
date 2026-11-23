#!/bin/bash
# Submits the full pooled weight PCA pipeline (Main Figure 5) as chained SLURM jobs:
#   1. pooled PCA of the encoding model weights, per ROI x feature space
#   2. PC interpretation: correlate each pooled PC time series with each annotated
#      feature (speech TRs only), with a 10,000-draw circular-shift permutation null
#   3. compile all results + BH-FDR into analysis/PCInterpretation/compiled_pooled_results.csv
# Each feature space below joins the top 3 layers of a model with '+'; they are
# concatenated along the feature axis into a single PCA.
# Usage: bash scripts/submit/submit_pooled_pca_workflow.sh   (from the project root)

set -e
MODEL=vislang_gpt2_chunk120_eps0.3
POPULATION=NT
source scripts/config.sh

activate_env
python -m src.pooled_pca combinations --dir $PROJECT_DIR --model $MODEL --feature_groups \
	"motion" \
	"alexnet_eps0.3_layer5+alexnet_eps0.3_layer6+alexnet_eps0.3_layer7" \
	"hubert_chunk120_layer10+hubert_chunk120_layer11+hubert_chunk120_layer12" \
	"GPT2_chunk120_layer10+GPT2_chunk120_layer11+GPT2_chunk120_layer12"

# Interpretation runs once per pooled PCA (subject column is the placeholder 'pooled')
PCA_COMBOS=$PROJECT_DIR/analysis/PooledPCA/pooled_pca_combinations.tsv
INTERP_COMBOS=$PROJECT_DIR/analysis/PCInterpretation/pooled_combinations.tsv
mkdir -p $(dirname $INTERP_COMBOS)
awk -F'\t' -v OFS='\t' '{print "pooled", $3, $1, $2}' $PCA_COMBOS > $INTERP_COMBOS
N=$(wc -l < $PCA_COMBOS)

PCA_JOB=$(sbatch --parsable --array=1-${N} scripts/sbatch/sbatch_pooled_pca.sh $MODEL $POPULATION)
INTERP_JOB=$(sbatch --parsable --dependency=afterok:$PCA_JOB --array=1-${N}%20 \
	scripts/sbatch/sbatch_pc_interpretation_pooled.sh $MODEL $POPULATION)
COMPILE_JOB=$(sbatch --parsable --dependency=afterok:$INTERP_JOB --partition=shared --mem=8G --time=0:30:00 \
	-o sbatch_logs/output_compile_pooled_%j.log -e sbatch_logs/error_compile_pooled_%j.log \
	--wrap="source scripts/config.sh; activate_env; python -u -m src.pc_interpretation compile --dir $PROJECT_DIR")

echo "Submitted $N ROI x feature combinations"
echo "  pooled PCA:         $PCA_JOB"
echo "  PC interpretation:  $INTERP_JOB (after $PCA_JOB)"
echo "  compile + FDR:      $COMPILE_JOB (after $INTERP_JOB)"
