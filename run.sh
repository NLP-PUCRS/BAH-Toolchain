#!/usr/bin/env bash
# ==============================================================================
# BAH-Toolchain Evaluation Runner
# ==============================================================================
# This script configures and executes the machine learning evaluation pipeline
# (scripts/evaluate.py) for Speech Emotion Recognition (SER) and acoustic classification.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# ------------------------------------------------------------------------------
# 1. Pipeline Configuration Variables
# ------------------------------------------------------------------------------

# FUSION: Type of fusion strategy
#   - 'early': Feature-level concatenation across representations
#   - 'late' : Decision-level fusion (saves predictions for soft-voting ensemble)
FUSION='late' # early, late

# MODE: Evaluation protocol / validation splitting scheme
#   - 'inner'          : Stratified 80/20 train/test split within the same dataset
#   - 'cross'          : Cross-corpus evaluation (train on DATASET, test on OTHER_DATA)
#   - 'multi'          : Multi-corpus evaluation (pools multiple datasets)
#   - 'inner-speaker'  : Speaker-independent split (leave-speakers-out)
#   - 'inner-sentence' : Sentence-independent split (leave-sentences-out)
MODE='inner' # cross, multi, inner

# CONCAT: Feature concatenation toggle
#   - False: Evaluate each feature representation independently (unimodal)
#   - True : Concatenate all features into a single vector prior to training (early fusion)
CONCAT=False

# CONF: Dataset and feature representation settings
DATASET="verbo"

# Target dataset for cross-corpus evaluation (required if MODE='cross')
OTHER_DATA=""

# Feature representations to evaluate:
FEATURES=(
    'hubert'
    'whisper'
    'wav2vec2'
    'wavlm'
    'frill'
    'eGeMAPSv02_88'
    'pAA'
    'vggish'
    'trillsson5'
    'ComParE_2016_6k'
    'trill'
)

# Output CSV file to log evaluation results
OUTPUT_CSV="eva_late_bah.csv"

# Random seeds for repeated evaluation
SEEDS=(42 97 123 2025)

# ------------------------------------------------------------------------------
# 2. CLI Argument Overrides (Optional)
# ------------------------------------------------------------------------------
# You can override the variables above via command-line arguments if desired:
#   ./run.sh --fusion early --concat true --mode cross --other-data crema-d
# ------------------------------------------------------------------------------
print_help() {
    cat << EOF
Usage: ./run.sh [OPTIONS]

Options:
  --fusion <early|late>       Fusion strategy (default: $FUSION)
  --mode <inner|cross|multi|inner-speaker|inner-sentence>
                              Evaluation protocol (default: $MODE)
  --concat <true|false>       Concatenate all features (default: $CONCAT)
  --dataset <name>            Primary dataset name (default: $DATASET)
  --other-data <name>         Target dataset for cross-corpus (default: '$OTHER_DATA')
  --features <f1 f2 ...>      List of features to evaluate
  --output-csv <filename>     Results CSV output file (default: $OUTPUT_CSV)
  -h, --help                  Show this help message and exit

Variables can also be edited directly inside run.sh.
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fusion)
            FUSION="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --concat)
            CONCAT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --other-data)
            OTHER_DATA="$2"
            shift 2
            ;;
        --output-csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --features)
            shift
            FEATURES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FEATURES+=("$1")
                shift
            done
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            ;;
    esac
done

# ------------------------------------------------------------------------------
# 3. Environment Resolution
# ------------------------------------------------------------------------------
# Resolve project root directory regardless of invocation location
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Detect Python interpreter (prefer conda environment 'admodel' if installed)
PYTHON_CMD="python3"
if command -v conda &> /dev/null; then
    if conda env list | grep -q "\bbah\b"; then
        echo "[INFO] Detected conda environment 'bah'. Executing via conda run..."
        PYTHON_CMD="conda run --no-capture-output -n bah python"
    elif conda env list | grep -q "\badmodel\b"; then
        echo "[INFO] Detected conda environment 'admodel'. Executing via conda run..."
        PYTHON_CMD="conda run --no-capture-output -n admodel python"
    fi
fi

# Ensure PYTHONPATH contains project root and model directory
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/model:${PYTHONPATH}"

# ------------------------------------------------------------------------------
# 4. Build Command Arguments & Execute
# ------------------------------------------------------------------------------
PY_ARGS=(
    "scripts/evaluate.py"
    "--fusion" "$FUSION"
    "--mode" "$MODE"
    "--dataset" "$DATASET"
    "--output-csv" "$OUTPUT_CSV"
    "--seeds" "${SEEDS[@]}"
)

# Handle CONCAT boolean flag
if [[ "$CONCAT" == "true" || "$CONCAT" == "True" || "$CONCAT" == "1" ]]; then
    PY_ARGS+=("--concat")
else
    PY_ARGS+=("--no-concat")
fi

# Add target dataset if specified for cross-corpus evaluation
if [[ -n "$OTHER_DATA" ]]; then
    PY_ARGS+=("--other-data" "$OTHER_DATA")
fi

# Add feature representations
PY_ARGS+=("--features" "${FEATURES[@]}")

echo "=============================================================================="
echo "BAH-Toolchain Evaluation Pipeline"
echo "=============================================================================="
echo "  Project Root : $PROJECT_ROOT"
echo "  Interpreter  : $PYTHON_CMD"
echo "  Fusion       : $FUSION"
echo "  Mode         : $MODE"
echo "  Concat       : $CONCAT"
echo "  Dataset      : $DATASET"
if [[ -n "$OTHER_DATA" ]]; then
    echo "  Other Data   : $OTHER_DATA"
fi
echo "  Features (${#FEATURES[@]}): ${FEATURES[*]}"
echo "  Output CSV   : $OUTPUT_CSV"
echo "  Seeds        : ${SEEDS[*]}"
echo "=============================================================================="

# Run evaluate.py
$PYTHON_CMD "${PY_ARGS[@]}"

