#!/usr/bin/env bash
# NAVER LABS large-indoor (B1, B2). Strict 0.25m / 2deg. Each building is a single reference_root
# with queries/{images,gt_poses.txt}; the sigma_joint gate is calibrated cross-building (B1<->B2).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Per-building reference roots (override for your machine).
B1_ROOT="${NAVER_B1_ROOT:-$RICLOC_REPO_ROOT/result/naver_gangnam_b1/reference_root}"
B2_ROOT="${NAVER_B2_ROOT:-$RICLOC_REPO_ROOT/result/naver_gangnam_b2/reference_root}"
OUT="$OUT_ROOT/naver"

print_header

# localize each building
eval_scene "naver" "$B1_ROOT" "$B1_ROOT/queries/images" "$B1_ROOT/queries/gt_poses.txt" "$OUT/b1" "$@"
eval_scene "naver" "$B2_ROOT" "$B2_ROOT/queries/images" "$B2_ROOT/queries/gt_poses.txt" "$OUT/b2" "$@"

# cross-building gate (calibrate on the other building's results)
if [[ -f "$OUT/b1/results.json" && -f "$OUT/b2/results.json" ]]; then
  echo; echo "==== NAVER cross-building held-out sigma_joint gate ===="
  "$PY" "$AGGREGATE" --glob "$OUT/b1/results.json" --dataset naver --calib_glob "$OUT/b2/results.json" --tag "B1|calib=B2"
  "$PY" "$AGGREGATE" --glob "$OUT/b2/results.json" --dataset naver --calib_glob "$OUT/b1/results.json" --tag "B2|calib=B1"
fi
