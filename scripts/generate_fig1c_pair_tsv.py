"""Emit the list of feature-space pairs (one per line, ``f1-f2``) for a
given encoding model or for the 6 Fig. 1C groups. Used as the SLURM array
input for the non-CV and no-chunk-CV controls.

Usage:
  # Fig 1C 6 groups (image, motion, social, speech, word, multi-sentence)
  python scripts/generate_fig1c_pair_tsv.py \
    --output scripts/featurespace_comparisons_fig1c_pairs.tsv

  # Every within-model pair for an encoding-model name
  python scripts/generate_fig1c_pair_tsv.py \
    --model joint_chunk120 \
    --output scripts/featurespace_comparisons_joint_chunk120.tsv
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cca_similarity_noncv_control import (
    FIG1C_GROUPS, FIG1C_GROUP_ORDER, get_model_feature_list,
)


def enumerate_fig1c_group_pairs():
    pairs = []
    for i, g1 in enumerate(FIG1C_GROUP_ORDER):
        for j, g2 in enumerate(FIG1C_GROUP_ORDER):
            if j <= i:
                continue
            for f1 in FIG1C_GROUPS[g1]:
                for f2 in FIG1C_GROUPS[g2]:
                    if f1 != f2:
                        pairs.append(f'{f1}-{f2}')
    return pairs


def enumerate_model_pairs(model_name):
    features = get_model_feature_list(model_name)
    pairs = []
    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            if f1 != f2:
                pairs.append(f'{f1}-{f2}')
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None,
                   help='If set, emit every within-model pair; otherwise '
                        'emit the 6 Fig. 1C groups pairs.')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    pairs = (enumerate_model_pairs(args.model) if args.model
             else enumerate_fig1c_group_pairs())
    with open(args.output, 'w') as fh:
        fh.write('\n'.join(pairs) + '\n')
    print(f'wrote {len(pairs)} pairs to {args.output}')


if __name__ == '__main__':
    main()
