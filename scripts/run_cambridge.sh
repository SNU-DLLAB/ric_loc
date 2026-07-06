#!/usr/bin/env bash
# Cambridge Landmarks (strict 0.25m / 2deg). Layout mirrors 7-Scenes.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-$DATA_ROOT/cambridge/processed}"
SCENES="${SCENES:-greatcourt kingscollege oldhospital shopfacade stmaryschurch}"
QUERY_SUBDIR="${QUERY_SUBDIR:-queries_full}"
OUT="$OUT_ROOT/cambridge"

print_header
for s in $SCENES; do
  ref="$CAMBRIDGE_ROOT/$s"
  eval_scene "cambridge" "$ref" "$ref/$QUERY_SUBDIR/images" "$ref/$QUERY_SUBDIR/gt_poses.txt" "$OUT/$s" "$@"
done

echo; echo "==== pooled Cambridge (leave-one-scene-out sigma_joint gate) ===="
"$PY" "$AGGREGATE" --glob "$OUT/*/results.json" --dataset cambridge --loso_key scene
