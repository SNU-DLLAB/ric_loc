#!/usr/bin/env bash
# 7-Scenes reference-count (K) sensitivity sweep.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

KSWEEP="${KSWEEP:-4 6 8 12 16}"
SEVENSCENES_ROOT="${SEVENSCENES_ROOT:-$DATA_ROOT/7scenes/processed}"
SCENES7="${SCENES7:-chess fire heads office pumpkin redkitchen stairs}"
QUERY_SUBDIR="${QUERY_SUBDIR:-queries_full}"
LIMIT="${LIMIT:-0}"                     # per-scene query cap (0 = all)

print_header
echo "ksweep=[$KSWEEP] limit=$LIMIT out=$OUT_ROOT/ksweep"

for K in $KSWEEP; do
  echo; echo "################# K=$K #################"
  for s in $SCENES7; do
    ref="$SEVENSCENES_ROOT/$s"
    eval_scene "7scenes" "$ref" "$ref/$QUERY_SUBDIR/images" "$ref/$QUERY_SUBDIR/gt_poses.txt" \
      "$OUT_ROOT/ksweep/K$K/7scenes/$s" --num_refs "$K" \
      $( [[ "$LIMIT" -gt 0 ]] && printf -- '--limit %s' "$LIMIT" )
  done
  echo "---- K=$K tables ----"
  "$PY" "$PUB_DIR/analyze_tables.py" --root "$OUT_ROOT/ksweep/K$K" \
    --gt7 "$SEVENSCENES_ROOT" --only 7scenes
done
