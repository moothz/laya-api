#!/usr/bin/env bash
# ==============================================================================
# Laya API - Inicialização em CPU (AMD Ryzen 9 9950X / Zen 5)
# Otimizado: 4 Workers Uvicorn + 4 Threads PyTorch + Taskset CCD 0 (0-7)
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

# Força execução exclusiva em CPU
export CUDA_VISIBLE_DEVICES=""
export LAYA_DEVICE="cpu"

# Otimizações de Threads Zen 5 para o Ryzen 9950X (4 threads por worker)
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export OPENBLAS_NUM_THREADS="4"
export LAYA_THREADS="4"
export LAYA_CPU_AMP="bf16"

export LAYA_MAX_LOADED="1"
export LAYA_PRELOAD_MODELS="multilingual"
export LAYA_DEFAULT_MODEL="multilingual"
export LAYA_ENABLE_VISION="1"

# Diretório de cache dos modelos Hugging Face
export HF_HOME="/mnt/data/services/laya/cache"

# Configurações de Rede e Servidor
export PORT="${PORT:-8005}"
export HOST="${HOST:-0.0.0.0}"
export WORKERS="${WORKERS:-4}"

echo "[*] Iniciando Laya API em CPU com taskset CCD0 (Porta: $PORT, Workers: $WORKERS)..."
# Fixa execução no primeiro CCD (núcleos 0 a 7) para evitar latência entre chiplets
exec taskset -c 0-7 python -m uvicorn server:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
