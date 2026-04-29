#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "Venv pronta em: $VENV_DIR"
echo "Ative com: source $VENV_DIR/bin/activate"
