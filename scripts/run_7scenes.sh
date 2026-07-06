#!/usr/bin/env bash
# 7-Scenes (strict 0.05m / 5deg). Layout: $SEVENSCENES_ROOT/<scene>/{dense/...,queries_full/{images,gt_poses.txt}}
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SEVENSCENES_ROOT="${SEVENSCENES_ROOT:-$DATA_ROOT/7scenes/processed}"
SCENES="${SCENES:-chess fire heads office pumpkin redkitchen stairs}"
QUERY_SUBDIR="${QUERY_SUBDIR:-queries_full}"     # use 'queries' for the quick subsampled split
OUT="$OUT_ROOT/7scenes"

print_header
for s in $SCENES; do
  ref="$SEVENSCENES_ROOT/$s"
  eval_scene "7scenes" "$ref" "$ref/$QUERY_SUBDIR/images" "$ref/$QUERY_SUBDIR/gt_poses.txt" "$OUT/$s" "$@"
done

echo; echo "==== pooled 7-Scenes (leave-one-scene-out sigma_joint gate) ===="
"$PY" "$AGGREGATE" --glob "$OUT/*/results.json" --dataset 7scenes --loso_key scene
