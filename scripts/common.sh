#!/usr/bin/env bash
# Shared environment + helpers for the RIC-Loc benchmark scripts.
set -euo pipefail

# repo / package locations
PUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # .../pub
REPO_ROOT_DEFAULT="$(cd "$PUB_DIR/.." && pwd)"                       # tree providing thirdparty/vggt + models/
export RICLOC_REPO_ROOT="${RICLOC_REPO_ROOT:-$REPO_ROOT_DEFAULT}"
# VGGT source tree (the importable vggt package)
export RICLOC_VGGT_SRC="${RICLOC_VGGT_SRC:-$RICLOC_REPO_ROOT/thirdparty/vggt}"
export RICLOC_MODELS_DIR="${RICLOC_MODELS_DIR:-$RICLOC_REPO_ROOT/models}"

# runtime knobs
PY="${PY:-python}"
export VGGT_CKPT="${VGGT_CKPT:-$RICLOC_MODELS_DIR/vggt_1B.pt}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# disable VGGT forward-output cache reuse (set 0 to re-enable the on-disk cache)
export UNIFIED_VGGT_DISABLE_CACHE="${UNIFIED_VGGT_DISABLE_CACHE:-1}"

# dataset roots (override for your machine)
export DATA_ROOT="${DATA_ROOT:-/media/dllab/HDD/data}"
export OUT_ROOT="${OUT_ROOT:-$PUB_DIR/outputs}"

EVALUATE="$PUB_DIR/evaluate.py"
AGGREGATE="$PUB_DIR/aggregate.py"

# eval one scene: eval_scene <dataset> <reference_root> <query_dir> <gt_file> <out_dir> [extra args...]
eval_scene() {
  local dataset="$1" ref="$2" qdir="$3" gt="$4" outdir="$5"; shift 5
  if [[ ! -d "$ref" ]]; then echo "[skip] reference_root missing: $ref"; return 0; fi
  if [[ ! -d "$qdir" ]]; then echo "[skip] query dir missing: $qdir"; return 0; fi
  echo "==== [$dataset] $(basename "$ref") ===="
  "$PY" "$EVALUATE" \
    --reference_root "$ref" --vggt_ckpt "$VGGT_CKPT" \
    --query_path "$qdir" ${gt:+--gt "$gt"} --dataset "$dataset" \
    --out_dir "$outdir" "$@"
}

print_header() {
  echo "RIC-Loc frozen benchmark | repo=$RICLOC_REPO_ROOT vggt_src=$RICLOC_VGGT_SRC ckpt=$VGGT_CKPT"
  echo "data_root=$DATA_ROOT out_root=$OUT_ROOT python=$($PY -c 'import sys;print(sys.executable)')"
}
