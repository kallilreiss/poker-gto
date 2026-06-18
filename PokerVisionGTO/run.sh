#!/usr/bin/env bash
# run.sh — Start Poker Vision GTO (API + Streamlit)

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Install dependencies ──────────────────────────────────────────────────────
if ! python -c "import fastapi" 2>/dev/null; then
  echo "[INFO] Instalando dependências…"
  pip install -r requirements.txt --quiet
fi

# ── Start FastAPI in background ───────────────────────────────────────────────
echo "[INFO] Iniciando FastAPI na porta 8000…"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
echo "[INFO] API PID: $API_PID"

sleep 2

# ── Start Streamlit ───────────────────────────────────────────────────────────
echo "[INFO] Iniciando Streamlit na porta 8501…"
streamlit run web/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true

# Cleanup on exit
kill $API_PID 2>/dev/null || true
