#!/usr/bin/env bash
# =============================================================================
# download_models.sh
#
# Downloads MoME+ expert checkpoints and fine-tuned LLM adapter from
# Google Drive onto the deployment server.
#
# USAGE:
#   1. Upload your model files to Google Drive and get the shareable links.
#   2. For each link (e.g. https://drive.google.com/file/d/FILE_ID/view),
#      copy the FILE_ID and replace the placeholders below.
#   3. Run from the repo root on the server:
#        bash scripts/download_models.sh
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Ensure gdown is available ─────────────────────────────────────────────────
if ! command -v gdown &>/dev/null; then
    info "Installing gdown..."
    pip install -q gdown --break-system-packages
fi

# ── Google Drive File IDs ─────────────────────────────────────────────────────
# Replace each <FILE_ID_...> with the actual ID from your Google Drive share URL.
# Example URL: https://drive.google.com/file/d/1A2B3C4D5E6F/view?usp=sharing
#              → File ID = 1A2B3C4D5E6F

EXPERT_T1_ID="1zjgJA2T-ubh7FdYu0vQsQHSt7QCeEZRs"
EXPERT_T1CE_ID="145uckiI0PjX5CciDl-fHjbRnW_G0oO0n"
EXPERT_T2_ID="1CJKplGT-j4-gG2PVR6PGGzraG_OpElW4"
EXPERT_FLAIR_ID="1QTWSpfolU3DEMhtScIeCs5-JnVpl8g2e"
MOME_FUSION_ID="16knEN0uDXUD-xtD54pbvJf6FiKh0MQzp"

# LLM LoRA adapter files
LLM_ADAPTER_CONFIG_ID="1JC-nJVmUiVkpaw75I-4Z_EJM2pjPRQAq"
LLM_ADAPTER_WEIGHTS_ID="11xi-jgruPqN8D39o7rJbvmn8B1dSTvzM"
LLM_TOKENIZER_ID="1MlU-Hqx9SqQdkPtK1Rk-41vOGuoyDtCx"
LLM_TOKENIZER_CONFIG_ID="1U7s3zDPLI2_DMHWRG7dxiPIFq9Vr9JPC"
LLM_CHAT_TEMPLATE_ID="1P-D1975a6aebMncWFWCZpKGol1-doZCw"
LLM_TRAINING_ARGS_ID="13rZrdSG3t4uvmJ7TkgRIZB24jXh8hWMA"

# ── Create target directories ─────────────────────────────────────────────────
info "Creating model directories..."
mkdir -p models/checkpoints/experts
mkdir -p models/report_generator/final

# ── Download segmentation expert checkpoints ──────────────────────────────────
info "Downloading MoME+ expert checkpoints (~1.6 GB total)..."

download_if_missing() {
    local dest="$1"
    local file_id="$2"
    local label="$3"

    if [[ "$file_id" == "<"*">" ]]; then
        warn "Skipping $label — FILE_ID not set. Edit scripts/download_models.sh first."
        return
    fi

    if [[ -f "$dest" ]]; then
        info "$label already exists, skipping."
        return
    fi

    info "Downloading $label..."
    gdown "https://drive.google.com/uc?id=${file_id}" -O "$dest" \
        || error "Failed to download $label (id=$file_id). Check the file is shared publicly."
}

download_if_missing "models/checkpoints/experts/expert_T1_best.pth"   "$EXPERT_T1_ID"   "expert_T1_best.pth"
download_if_missing "models/checkpoints/experts/expert_T1ce_best.pth" "$EXPERT_T1CE_ID" "expert_T1ce_best.pth"
download_if_missing "models/checkpoints/experts/expert_T2_best.pth"   "$EXPERT_T2_ID"   "expert_T2_best.pth"
download_if_missing "models/checkpoints/experts/expert_FLAIR_best.pth" "$EXPERT_FLAIR_ID" "expert_FLAIR_best.pth"
download_if_missing "models/checkpoints/experts/mome_fusion_best.pth" "$MOME_FUSION_ID" "mome_fusion_best.pth"

# ── Download fine-tuned LLM adapter ───────────────────────────────────────────
info "Downloading fine-tuned LLM LoRA adapter..."
download_if_missing "models/report_generator/final/adapter_config.json"         "$LLM_ADAPTER_CONFIG_ID"  "adapter_config.json"
download_if_missing "models/report_generator/final/adapter_model.safetensors"   "$LLM_ADAPTER_WEIGHTS_ID" "adapter_model.safetensors"
download_if_missing "models/report_generator/final/tokenizer.json"              "$LLM_TOKENIZER_ID"       "tokenizer.json"
download_if_missing "models/report_generator/final/tokenizer_config.json"       "$LLM_TOKENIZER_CONFIG_ID" "tokenizer_config.json"
download_if_missing "models/report_generator/final/chat_template.jinja"         "$LLM_CHAT_TEMPLATE_ID"   "chat_template.jinja"
download_if_missing "models/report_generator/final/training_args.bin"           "$LLM_TRAINING_ARGS_ID"   "training_args.bin"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
info "✅ Model download complete. Directory contents:"
echo ""
find models/checkpoints/experts/ models/report_generator/final/ -type f \
    -exec ls -lh {} \; 2>/dev/null || true
echo ""
info "You can now run: docker compose up -d --build"
