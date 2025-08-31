#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/server"
mkdir -p certificates
cd certificates
if [[ ! -f key.pem || ! -f cert.pem ]]; then
  openssl req -x509 -newkey rsa:4096 -nodes -sha256 -days 3650     -keyout key.pem -out cert.pem     -subj "/CN=localhost"     -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0.0.0.0"
  echo "Self-signed certificate generated in $(pwd)"
else
  echo "Certificates already exist in $(pwd)"
fi
