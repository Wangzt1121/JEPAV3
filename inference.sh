#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/A432/wzt/robort/le-wm"
CACHE_DIR="/A432/wzt/.stable_worldmodel"
# SOURCE_DIR="$CACHE_DIR/hf_pusht"
# TARGET_DIR="$CACHE_DIR/checkpoints/models--quentinll--lewm-pusht"
# TEST_DATA="/A432/wzt/robort/dataset/pusht_expert_test_10"

# SOURCE_DIR="/A432/wzt/.stable_worldmodel/hf_lewm-cube"
# TARGET_DIR="/A432/wzt/.stable_worldmodel/checkpoints/models--quentinll--lewm-cube"
# TEST_DATA="/A432/wzt/robort/dataset/cube_single_expert_test_10"
# CONFIG_NAME=cube.yaml
# POLICY=quentinll/lewm-cube

# SOURCE_DIR="/A432/wzt/.stable_worldmodel/hf_lewm-reacher"
# TARGET_DIR="/A432/wzt/.stable_worldmodel/checkpoints/models--quentinll--lewm-reacher"
# TEST_DATA="/A432/wzt/robort/dataset/reacher_test_10"
# POLICY=quentinll/lewm-reacher
# CONFIG_NAME=reacher.yaml

SOURCE_DIR="/A432/wzt/.stable_worldmodel/hf_lewm-tworooms"
TARGET_DIR="/A432/wzt/.stable_worldmodel/checkpoints/models--quentinll--lewm-tworooms"
TEST_DATA="/A432/wzt/robort/dataset/tworoom_test_10"
POLICY=quentinll/lewm-tworooms
CONFIG_NAME=tworoom.yaml

NUM_EVAL="${NUM_EVAL:-50}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"

export STABLEWM_HOME="$CACHE_DIR"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi

if [[ ! -f "$SOURCE_DIR/config.json" || ! -f "$SOURCE_DIR/weights.pt" ]]; then
    echo "Official PushT checkpoint is missing from $SOURCE_DIR" >&2
    echo "Expected config.json and weights.pt" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"
cp "$SOURCE_DIR/config.json" "$TARGET_DIR/config.json"

tmp_weights="$TARGET_DIR/weights.pt.tmp.$$"
trap 'rm -f "$tmp_weights"' EXIT

"$PYTHON" - "$SOURCE_DIR/weights.pt" "$tmp_weights" <<'PY'
from pathlib import Path
import sys

import torch

source = Path(sys.argv[1])
target = Path(sys.argv[2])
state_dict = torch.load(source, map_location="cpu", weights_only=False)


def convert_key(key):
    if not key.startswith("encoder.encoder.layer."):
        return key

    key = key.replace("encoder.encoder.layer.", "encoder.layers.")
    replacements = {
        ".attention.attention.query.": ".attention.q_proj.",
        ".attention.attention.key.": ".attention.k_proj.",
        ".attention.attention.value.": ".attention.v_proj.",
        ".attention.output.dense.": ".attention.o_proj.",
        ".intermediate.dense.": ".mlp.fc1.",
        ".output.dense.": ".mlp.fc2.",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return key


converted = {convert_key(key): value for key, value in state_dict.items()}
torch.save(converted, target)
print(f"converted {len(converted)} parameters from {source}")
PY

mv -f "$tmp_weights" "$TARGET_DIR/weights.pt"
trap - EXIT

echo "Official PushT checkpoint is ready: $TARGET_DIR"
echo "Evaluation dataset: ${TEST_DATA}.h5"
echo "num_eval=$NUM_EVAL goal_offset_steps=$GOAL_OFFSET_STEPS eval_budget=$EVAL_BUDGET"

CUDA_VISIBLE_DEVICES=1,2 \
    "$PYTHON" eval.py \
    --config-name="$CONFIG_NAME" \
    policy="$POLICY" \
    "eval.dataset_name=$TEST_DATA" \
    "eval.num_eval=$NUM_EVAL" \
    "eval.goal_offset_steps=$GOAL_OFFSET_STEPS" \
    "eval.eval_budget=$EVAL_BUDGET"
