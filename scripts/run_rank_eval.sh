#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_VENV_PYTHON="${REPO_ROOT}/venv/bin/python"

if [[ -n "${VENV_PYTHON:-}" ]]; then
    PYTHON_BIN="${VENV_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
else
    PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Expected Python interpreter not found at ${PYTHON_BIN:-<empty>}" >&2
    echo "Activate your virtual environment first, set VENV_PYTHON=/path/to/python, or create ${DEFAULT_VENV_PYTHON}." >&2
    exit 1
fi

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

EVAL_DATASETS_FOLDER="${EVAL_DATASETS_FOLDER:-datasets}"
DATASETS=(${DATASETS:-msls sped})
MODELS=(${MODELS:-checkpoints/SuperVLAD_adverserially_trained.pth})
MODEL_TAGS=(${MODEL_TAGS:-trained})
FOUNDATION_MODEL_PATH="${FOUNDATION_MODEL_PATH:-checkpoints/dinov2_vitb14_pretrain.pth}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-8}"
BACKBONE="${BACKBONE:-dino}"
SUPERVLAD_CLUSTERS="${SUPERVLAD_CLUSTERS:-4}"
RANK_ATTACK="${RANK_ATTACK:-rank_pgd_linf}"
RANK_STEPS="${RANK_STEPS:-20}"
RANK_RESTARTS="${RANK_RESTARTS:-1}"
BATCH_ID="${BATCH_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
EPSILONS=(${EPSILONS:-0.01 0.1})
EXTRA_ARGS=("$@")

for model_path in "${MODELS[@]}"; do
    if [[ ! -f "${model_path}" ]]; then
        echo "Checkpoint not found: ${model_path}" >&2
        exit 1
    fi
done

if [[ ! -f "${FOUNDATION_MODEL_PATH}" ]]; then
    echo "Foundation model checkpoint not found: ${FOUNDATION_MODEL_PATH}" >&2
    exit 1
fi

OUTPUT_DIR="test/rank_eval/${BATCH_ID}/${RANK_ATTACK}"
mkdir -p "${OUTPUT_DIR}"

MODEL_TAG_ARGS=()
if (( ${#MODEL_TAGS[@]} > 0 )); then
    MODEL_TAG_ARGS=(--model_tags "${MODEL_TAGS[@]}")
fi

"${PYTHON_BIN}" "${REPO_ROOT}/rank_eval.py" \
    --eval_datasets_folder="${EVAL_DATASETS_FOLDER}" \
    --datasets "${DATASETS[@]}" \
    --model_type=supervlad \
    --model_paths "${MODELS[@]}" \
    "${MODEL_TAG_ARGS[@]}" \
    --foundation_model_path="${FOUNDATION_MODEL_PATH}" \
    --backbone="${BACKBONE}" \
    --supervlad_clusters="${SUPERVLAD_CLUSTERS}" \
    --crossimage_encoder \
    --infer_batch_size="${INFER_BATCH_SIZE}" \
    --rank_attack="${RANK_ATTACK}" \
    --rank_steps="${RANK_STEPS}" \
    --rank_restarts="${RANK_RESTARTS}" \
    --epsilons "${EPSILONS[@]}" \
    --output_json="${OUTPUT_DIR}/rank_eval_results.json" \
    --output_csv="${OUTPUT_DIR}/rank_eval_results.csv" \
    "${EXTRA_ARGS[@]}"

echo "Finished rank evaluation. Results are under ${OUTPUT_DIR}/"
