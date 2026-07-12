#!/usr/bin/env bash
set -euo pipefail

AUTODL_PORT="${AUTODL_PORT:-6006}"
APP_PORT="${SP2026_PORT:-8000}"
CONFIG_DIR="${HOME}/summer2026-config"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== SummerProject2026 AutoDL Deployment ==="

echo "1. Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg libgl1 libglib2.0-0

echo "2. Setting up config..."
mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    cp "$PROJECT_DIR/config-template.yaml" "$CONFIG_DIR/config.yaml"
    echo "   Edit $CONFIG_DIR/config.yaml to set your cameras and database."
fi

if [ ! -f "$CONFIG_DIR/.env" ]; then
    cat > "$CONFIG_DIR/.env" <<EOF
SP2026_DEBUG=false
SP2026_HOST=0.0.0.0
SP2026_PORT=${APP_PORT}
SP2026_ENABLE_STREAMING=true
SP2026_ENABLE_TRAFFIC_ANALYST=true
SP2026_CORS_ORIGINS=["*"]
EOF
    echo "   Created $CONFIG_DIR/.env"
fi

echo "3. Installing Python dependencies..."
cd "$PROJECT_DIR"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "4. Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "5. Starting API server..."
echo "   Public URL: https://connect.${AUTODL_PORT}.autodl.team"
echo "   Local URL:  http://127.0.0.1:${APP_PORT}"
echo ""

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "$PROJECT_DIR"
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "${APP_PORT}" \
    --workers 1 \
    --loop asyncio
