#!/usr/bin/env bash
set -euo pipefail

DES="$(hostname -f | cut -d. -f1)"
BASE_DIR="$HOME/process_listener"
SERVER_IP="172.20.83.202"

cd "$BASE_DIR"

mkdir -p "metrics" "tmp_data_dir_${DES}"

if screen -list | grep -q "[.]chrome[[:space:]]"; then
  echo "chrome screen already exists"
else
  screen -S chrome -dm bash -lc "
    google-chrome \
      --ozone-platform=headless \
      --headless \
      --enable-features=Vulkan \
      --disable-vulkan-surface \
      --disable-software-rasterizer \
      --ignore-gpu-blocklist \
      --use-angle=vulkan \
      --enable-unsafe-webgpu \
      --ignore-ssl-errors \
      --ignore-certificate-errors \
      --disable-gpu-sandbox \
      --disable-gpu-watchdog \
      --no-sandbox \
      --use-cmd-decoder=passthrough \
      --user-data-dir='$BASE_DIR/tmp_data_dir_${DES}/' \
      'https://${SERVER_IP}:3000/?mode=headless&workerId=${DES}&log=debug'
  "
  echo "started chrome"
fi

if screen -list | grep -q "[.]listener[[:space:]]"; then
  echo "listener screen already exists"
else
  screen -S listener -dm bash -lc "
    cd '$BASE_DIR' &&
    exec ./build/unified_monitor \
      --label ${DES} \
      --out-dir metrics/ \
      --interval 100 \
      --gpu-index 0
  "
  echo "started listener"
fi

echo okay
