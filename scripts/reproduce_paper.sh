#!/usr/bin/env bash
# Run the full RIC-Loc benchmark over all datasets and print the result tables.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-7scenes cambridge 12scenes naver}"
SEVENSCENES_ROOT="${SEVENSCENES_ROOT:-$DATA_ROOT/7scenes/processed}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-$DATA_ROOT/cambridge/processed}"
TWELVESCENES_ROOT="${TWELVESCENES_ROOT:-$DATA_ROOT/12scenes/processed}"
SCENES7="${SCENES7:-chess fire heads office pumpkin redkitchen stairs}"
SCENESC="${SCENESC:-greatcourt kingscollege oldhospital shopfacade stmaryschurch}"
SCENES12="${SCENES12:-apt1_kitchen apt1_living apt2_bed apt2_kitchen apt2_living apt2_luke office1_gates362 office1_gates381 office1_lounge office1_manolis office2_5a office2_5b}"
QUERY_SUBDIR="${QUERY_SUBDIR:-queries_full}"
B1_ROOT="${NAVER_B1_ROOT:-$RICLOC_REPO_ROOT/result/naver_gangnam_b1/reference_root}"
B2_ROOT="${NAVER_B2_ROOT:-$RICLOC_REPO_ROOT/result/naver_gangnam_b2/reference_root}"

# strip a leading `--` so extra evaluate.py args pass through
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")
extra() { ((${#EXTRA[@]})) && printf '%s\n' "${EXTRA[@]}"; }   # safe under set -u

want() { [[ " $DATASETS " == *" $1 "* ]]; }

print_header
echo "datasets=[$DATASETS]  out_root=$OUT_ROOT  query_subdir=$QUERY_SUBDIR  extra=[${EXTRA[*]:-}]"

# optional preflight on the first resolvable map
if [[ "${RUN_CHECK:-0}" == "1" ]]; then
  for r in "$SEVENSCENES_ROOT"/* "$CAMBRIDGE_ROOT"/* "$B1_ROOT" "$B2_ROOT"; do
    [[ -d "$r/dense" || -d "$r/sparse" ]] || continue
    echo "==== preflight setup_check: $r ===="
    "$PY" "$PUB_DIR/setup_check.py" --reference_root "$r" --vggt_ckpt "$VGGT_CKPT" || true
    break
  done
fi

# live runs
if want 7scenes; then
  for s in $SCENES7; do
    ref="$SEVENSCENES_ROOT/$s"
    eval_scene "7scenes" "$ref" "$ref/$QUERY_SUBDIR/images" "$ref/$QUERY_SUBDIR/gt_poses.txt" \
      "$OUT_ROOT/7scenes/$s" ${EXTRA[@]+"${EXTRA[@]}"}
  done
fi
if want cambridge; then
  for s in $SCENESC; do
    ref="$CAMBRIDGE_ROOT/$s"
    eval_scene "cambridge" "$ref" "$ref/$QUERY_SUBDIR/images" "$ref/$QUERY_SUBDIR/gt_poses.txt" \
      "$OUT_ROOT/cambridge/$s" ${EXTRA[@]+"${EXTRA[@]}"}
  done
fi
if want 12scenes; then
  # 12-Scenes uses queries/ (stride=1, already the full query split), not $QUERY_SUBDIR.
  for s in $SCENES12; do
    ref="$TWELVESCENES_ROOT/$s"
    eval_scene "12scenes" "$ref" "$ref/queries/images" "$ref/queries/gt_poses.txt" \
      "$OUT_ROOT/12scenes/$s" ${EXTRA[@]+"${EXTRA[@]}"}
  done
fi
if want naver; then
  eval_scene "naver" "$B1_ROOT" "$B1_ROOT/queries/images" "$B1_ROOT/queries/gt_poses.txt" \
    "$OUT_ROOT/naver/b1" ${EXTRA[@]+"${EXTRA[@]}"}
  eval_scene "naver" "$B2_ROOT" "$B2_ROOT/queries/images" "$B2_ROOT/queries/gt_poses.txt" \
    "$OUT_ROOT/naver/b2" ${EXTRA[@]+"${EXTRA[@]}"}
fi

# result tables from the freshly-produced dumps
echo; echo "################# PAPER TABLES #################"
ONLY=()
read -ra _ds <<< "$DATASETS"
[[ ${#_ds[@]} -eq 1 ]] && ONLY=(--only "${_ds[0]}")
GTARGS=(--gt7 "$SEVENSCENES_ROOT" --gtc "$CAMBRIDGE_ROOT" --gt12 "$TWELVESCENES_ROOT"
        --gtnb1 "$B1_ROOT/queries/gt_poses.txt" --gtnb2 "$B2_ROOT/queries/gt_poses.txt")
"$PY" "$PUB_DIR/analyze_tables.py" --root "$OUT_ROOT" "${GTARGS[@]}" "${ONLY[@]}"

echo; echo "################# SUPPLEMENTARY TABLES (live reproduction) #################"
"$PY" "$PUB_DIR/analyze_supp.py" --root "$OUT_ROOT" "${GTARGS[@]}" "${ONLY[@]}"
