#!/usr/bin/env bash
set -euo pipefail

AUTODL_PORT="${AUTODL_PORT:-6006}"
APP_PORT="${SP2026_PORT:-8000}"
TURN_USER="${TURN_USER:-summer}"
TURN_PASSWORD="${TURN_PASSWORD:-$(openssl rand -hex 8)}"
CONFIG_DIR="${HOME}/summer2026-config"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== SummerProject2026 AutoDL Deployment ==="
echo ""

echo "1. Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 netcat-openbsd

echo "2. Starting TURN server..."
docker rm -f coturn 2>/dev/null || true
docker run -d --name coturn --network host --restart unless-stopped \
  coturn/coturn:latest \
  -n --listening-port=3478 \
  --min-port=49152 --max-port=65535 \
  --fingerprint --lt-cred-mech \
  --user="${TURN_USER}:${TURN_PASSWORD}" \
  --realm=autodl \
  --log-file=stdout
echo "   TURN user: ${TURN_USER}"
echo "   TURN password: ${TURN_PASSWORD}"

echo "3. Setting up config..."
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
SP2026_ICE_SERVERS=stun:stun.l.google.com:19302;turn:127.0.0.1:3478,username=${TURN_USER},credential=${TURN_PASSWORD}
EOF
    echo "   Created $CONFIG_DIR/.env"
fi

echo "4. Installing Python dependencies..."
cd "$PROJECT_DIR"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "5. Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Deployment Complete ==="
echo "Public URL:  https://connect.${AUTODL_PORT}.autodl.team"
echo "TURN server: 127.0.0.1:3478 (user: ${TURN_USER}, pass: ${TURN_PASSWORD})"
echo ""
echo "Frontend vite.config.ts proxy target:"
echo "  proxy: { '/api': { target: 'https://connect.${AUTODL_PORT}.autodl.team', changeOrigin: true } }"
echo ""

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "$PROJECT_DIR"
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "${APP_PORT}" \
    --workers 1 \
    --loop asyncio
