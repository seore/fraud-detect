#!/usr/bin/env bash
# Setup script for Credit Card Fraud Detection project on macOS
# - Installs Python 3.10 (via Homebrew)
# - Creates a virtual environment in .venv
# - Installs dependencies from requirements.txt
# - Runs main.py using the new environment

set -e  # exit immediately on error

echo "========================================="
echo "  Credit Card Fraud Detection - Setup"
echo "========================================="

# Go to the directory where this script lives (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# -------------------------------
# 1. Check Homebrew
# -------------------------------
if ! command -v brew >/dev/null 2>&1; then
  echo "❌ Homebrew is not installed."
  echo "Please install Homebrew first from https://brew.sh/ and re-run this script."
  exit 1
fi

echo "✅ Homebrew found."

# -------------------------------
# 2. Install Python 3.10 & libomp
# -------------------------------
echo "Installing python@3.10 and libomp (for xgboost)..."
brew install python@3.10 libomp || true

PYTHON_BIN="$(brew --prefix python@3.10)/bin/python3.10"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Could not find python3.10 at $PYTHON_BIN"
  echo "Make sure python@3.10 installed correctly."
  exit 1
fi

echo "✅ Using Python: $("$PYTHON_BIN" --version)"

# -------------------------------
# 3. Create virtual environment
# -------------------------------
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "Virtual environment .venv already exists. Reusing it."
fi

# Activate venv for this script
# (won't persist after script finishes; you'll activate manually later)
# shellcheck source=/dev/null
source .venv/bin/activate

echo "✅ Virtual environment activated: $(python --version)"

# -------------------------------
# 4. Upgrade pip & install deps
# -------------------------------
echo "Upgrading pip..."
python -m pip install --upgrade pip

if [ ! -f "requirements.txt" ]; then
  echo "❌ requirements.txt not found in project root."
  deactivate || true
  exit 1
fi

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "✅ Dependencies installed."

# -------------------------------
# 5. Quick sanity check: tensorflow
# -------------------------------
python - << 'EOF'
try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully:", tf.__version__)
except Exception as e:
    print("⚠️  WARNING: TensorFlow could not be imported.")
    print("   You can still run the project without the autoencoder part.")
    print("   Error was:", e)
EOF

# -------------------------------
# 6. Run the main pipeline
# -------------------------------
echo "========================================="
echo " Running main.py with Python 3.10 (.venv)"
echo "========================================="

python main.py

echo "========================================="
echo " Done!"
echo "========================================="
echo ""
echo "To use this environment in a new terminal session:"
echo "  cd \"$PROJECT_ROOT\""
echo "  source .venv/bin/activate"
echo "Then you can run:"
echo "  python main.py"
