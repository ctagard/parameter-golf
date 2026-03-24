#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
echo "Downloading data..."
python3 data/cached_challenge_fineweb.py --variant sp1024
echo "Installing requirements..."
pip install -r records/track_10min_16mb/2026-03-24_LoopedGPT_mHCLite_3x2_TTT/requirements.txt
echo "Setup complete."
