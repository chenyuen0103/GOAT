#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create or update a Conda environment for GOAT.

Usage:
  bash scripts/setup_env.sh [options]

Options:
  --name NAME              Conda environment name (default: goat)
  --python VERSION         Python version for a new environment (default: 3.9)
  --minimal                Install requirements-minimal.txt instead of the full
                           experiment dependencies in requirements-gpu.txt
  --dev                    Also install requirements-dev.txt
  --test                   Run the repository test suite after installation
  --torch-index-url URL    Install PyTorch/Torchvision from this wheel index
                           before installing the selected requirements
  --project-root PATH      Create portable data/model/cache/output directories
                           and persist their paths in the Conda environment
  --dry-run                Print commands without changing an environment
  -h, --help               Show this help message

Environment variables:
  GOAT_CONDA_ENV            Alternative default for --name
  GOAT_PYTHON_VERSION       Alternative default for --python
  TORCH_INDEX_URL           Alternative to --torch-index-url
  GOAT_PROJECT_ROOT         Alternative to --project-root

Examples:
  bash scripts/setup_env.sh
  bash scripts/setup_env.sh --dev --test
  bash scripts/setup_env.sh --project-root /project/yuen_chen --dev --test
  TORCH_INDEX_URL=<matching-PyTorch-index> bash scripts/setup_env.sh --dev
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${GOAT_CONDA_ENV:-goat}"
PYTHON_VERSION="${GOAT_PYTHON_VERSION:-3.9}"
PROFILE="full"
INSTALL_DEV=0
RUN_TESTS=0
DRY_RUN=0
TORCH_WHEEL_INDEX="${TORCH_INDEX_URL:-}"
PROJECT_ROOT="${GOAT_PROJECT_ROOT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 ]] || { echo "error: --name requires a value" >&2; exit 2; }
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "error: --python requires a value" >&2; exit 2; }
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --minimal)
      PROFILE="minimal"
      shift
      ;;
    --dev)
      INSTALL_DEV=1
      shift
      ;;
    --test)
      RUN_TESTS=1
      INSTALL_DEV=1
      shift
      ;;
    --torch-index-url)
      [[ $# -ge 2 ]] || { echo "error: --torch-index-url requires a value" >&2; exit 2; }
      TORCH_WHEEL_INDEX="$2"
      shift 2
      ;;
    --project-root)
      [[ $# -ge 2 ]] || { echo "error: --project-root requires a value" >&2; exit 2; }
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: environment name may contain only letters, numbers, '.', '_', and '-'" >&2
  exit 2
fi
if [[ ! "$PYTHON_VERSION" =~ ^[0-9]+([.][0-9]+){1,2}$ ]]; then
  echo "error: invalid Python version: $PYTHON_VERSION" >&2
  exit 2
fi
if [[ -n "$PROJECT_ROOT" && "$PROJECT_ROOT" != /* ]]; then
  echo "error: --project-root must be an absolute path" >&2
  exit 2
fi
if [[ "$PROJECT_ROOT" != "/" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT%/}"
fi

if [[ "$PROFILE" == "minimal" ]]; then
  REQUIREMENTS_FILE="${REPO_ROOT}/requirements-minimal.txt"
  VERIFY_IMPORTS="import matplotlib, numpy, ot, pandas, scipy, sklearn, torch, torchvision"
else
  REQUIREMENTS_FILE="${REPO_ROOT}/requirements-gpu.txt"
  VERIFY_IMPORTS="import matplotlib, numpy, ot, pandas, scipy, sklearn, tensorflow, torch, torchvision"
fi

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

if [[ "$DRY_RUN" -eq 0 ]] && ! command -v conda >/dev/null 2>&1; then
  echo "error: conda was not found; install Miniconda/Anaconda or initialize Conda first" >&2
  exit 1
fi

ENV_EXISTS=0
if [[ "$DRY_RUN" -eq 0 ]] && conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  ENV_EXISTS=1
fi

if [[ "$ENV_EXISTS" -eq 0 ]]; then
  run conda create --yes --name "$ENV_NAME" "python=${PYTHON_VERSION}" pip
else
  echo "Using existing Conda environment: ${ENV_NAME}"
fi

if [[ -n "$PROJECT_ROOT" ]]; then
  DATA_ROOT="${PROJECT_ROOT}/data"
  MODEL_ROOT="${PROJECT_ROOT}/models"
  CACHE_ROOT="${PROJECT_ROOT}/cache"
  OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
  run mkdir -p \
    "${DATA_ROOT}/mnist" \
    "${DATA_ROOT}/portraits/dataset_32x32" \
    "${DATA_ROOT}/covtype" \
    "$MODEL_ROOT" \
    "${CACHE_ROOT}/goat" \
    "${CACHE_ROOT}/keras" \
    "${CACHE_ROOT}/prepared" \
    "${OUTPUT_ROOT}/logs" \
    "${OUTPUT_ROOT}/plots"
  run conda env config vars set --name "$ENV_NAME" \
    "GOAT_PROJECT_ROOT=${PROJECT_ROOT}" \
    "GOAT_DATA_DIR=${DATA_ROOT}" \
    "GOAT_MODEL_DIR=${MODEL_ROOT}" \
    "GOAT_CACHE_DIR=${CACHE_ROOT}/goat" \
    "GOAT_PREPARED_ARTIFACT_ROOT=${CACHE_ROOT}/prepared" \
    "GOAT_OUTPUT_DIR=${OUTPUT_ROOT}" \
    "LOG_ROOT=${OUTPUT_ROOT}/logs" \
    "PLOT_ROOT=${OUTPUT_ROOT}/plots" \
    "KERAS_HOME=${CACHE_ROOT}/keras"
fi

CONDA_PYTHON=(conda run --no-capture-output -n "$ENV_NAME" python)
run "${CONDA_PYTHON[@]}" -m pip install --upgrade pip setuptools wheel

if [[ -n "$TORCH_WHEEL_INDEX" ]]; then
  run "${CONDA_PYTHON[@]}" -m pip install \
    --index-url "$TORCH_WHEEL_INDEX" \
    "torch>=2.2" "torchvision>=0.17"
fi

run "${CONDA_PYTHON[@]}" -m pip install -r "$REQUIREMENTS_FILE"

if [[ "$INSTALL_DEV" -eq 1 ]]; then
  run "${CONDA_PYTHON[@]}" -m pip install -r "${REPO_ROOT}/requirements-dev.txt"
fi

run "${CONDA_PYTHON[@]}" -c "$VERIFY_IMPORTS"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  run conda run --no-capture-output -n "$ENV_NAME" \
    env MPLCONFIGDIR="${REPO_ROOT}/.mplconfig" \
    python -m pytest -q "$REPO_ROOT/tests"
fi

cat <<EOF

GOAT environment is ready.

Activate it with:
  conda activate ${ENV_NAME}

Then run commands from:
  ${REPO_ROOT}
EOF
