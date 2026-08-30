#!/usr/bin/env bash
# One-time setup for the Proxmox LXC demo container.
# Run as root on the container: bash /opt/ocr-cascade/demo/setup-demo.sh
set -e

APP_DIR="/opt/ocr-cascade"
APP_USER="ocr"

echo "=== OCR Cascade Demo Setup ==="

# 1. Create a dedicated system user if it doesn't exist
if ! id "$APP_USER" &>/dev/null; then
  echo "[1] Creating system user '$APP_USER'..."
  useradd --system --no-create-home --shell /bin/false "$APP_USER"
else
  echo "[1] User '$APP_USER' already exists."
fi

# 2. Set ownership on app directory (root owns git, ocr user runs the app)
echo "[2] Setting ownership of $APP_DIR..."
chown -R root:root "$APP_DIR"
# Allow the app user to write runs/ and temp dirs
mkdir -p "$APP_DIR/runs"
chown "$APP_USER":"$APP_USER" "$APP_DIR/runs"

# 3. Install Python venv (CPU-only — no --gpu flag)
echo "[3] Installing Python environment (CPU-only)..."
cd "$APP_DIR"
bash "$APP_DIR/setup.sh"
chown -R "$APP_USER":"$APP_USER" "$APP_DIR/venv"

# 4. Install systemd unit files
echo "[4] Installing systemd units..."
cp "$APP_DIR/demo/ocr-cascade.service"         /etc/systemd/system/
cp "$APP_DIR/demo/ocr-cascade-cleanup.service" /etc/systemd/system/
cp "$APP_DIR/demo/ocr-cascade-cleanup.timer"   /etc/systemd/system/
cp "$APP_DIR/demo/ocr-cascade-update.service"  /etc/systemd/system/
cp "$APP_DIR/demo/ocr-cascade-update.timer"    /etc/systemd/system/

systemctl daemon-reload

# 5. Start app service
systemctl enable --now ocr-cascade.service

# 6. Start hourly cleanup timer
systemctl enable --now ocr-cascade-cleanup.timer

# 7. Start hourly GitHub update timer
systemctl enable --now ocr-cascade-update.timer

echo ""
echo "=== Setup complete ==="
echo ""
echo "App running at: http://127.0.0.1:8765"
echo "Point nginx (or your reverse proxy) at that address."
echo ""
echo "Service status:"
systemctl status ocr-cascade --no-pager
echo ""
echo "Active timers:"
systemctl list-timers 'ocr-cascade-*.timer' --no-pager
