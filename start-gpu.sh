#!/usr/bin/env bash
# ==============================================================================
# Laya API - Inicialização em GPU (RTX 5090 / CUDA)
# Otimizado para baixo consumo de VRAM (< 1.5 GB) e alta velocidade
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Ativa o ambiente virtual
if [ -d "$DIR/.venv" ]; then
    source "$DIR/.venv/bin/activate"
elif [ -d "/mnt/data/services/laya/.venv" ]; then
    source "/mnt/data/services/laya/.venv/bin/activate"
fi

# Configurações de GPU
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LAYA_DEVICE="cuda"
export LAYA_CUDA_AMP="bf16"

# Mantém apenas 1 modelo de texto residente por vez em BF16 (~650MB VRAM)
export LAYA_MAX_LOADED="1"
export LAYA_PRELOAD_MODELS="multilingual"
export LAYA_DEFAULT_MODEL="multilingual"
export LAYA_ENABLE_VISION="1"

# Diretório de cache dos modelos Hugging Face
export HF_HOME="/mnt/data/services/laya/cache"

# Configurações de Rede e Servidor
export PORT="${PORT:-8002}"
export HOST="${HOST:-0.0.0.0}"

echo "[*] Iniciando Laya API em GPU (Porta: $PORT, Host: $HOST)..."
exec python -m uvicorn server:app --host "$HOST" --port "$PORT" --workers 1
