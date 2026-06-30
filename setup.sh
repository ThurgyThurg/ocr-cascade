#!/usr/bin/env bash
set -e

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"

GPU_MODE=false
for arg in "$@"; do
  [ "$arg" = "--gpu" ] && GPU_MODE=true
done

echo "=== OCR Pipeline Setup ==="
$GPU_MODE && echo "    GPU mode enabled (CUDA packages will be installed)"

# 1. Python venv
if [ ! -f venv/bin/activate ]; then
  echo "[1/5] Creating Python virtual environment..."
  python -m venv venv
else
  echo "[1/5] venv already exists, skipping."
fi

source venv/bin/activate

# 2. Install Python packages
# Note: paddleocr is pinned <3.0 because v3.x triggers
# "NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support"
# at predict() time on the current paddlepaddle release.
echo "[2/5] Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet \
  "pymupdf>=1.24" \
  "pillow>=10.0" \
  "opencv-python-headless>=4.9" \
  "pytesseract>=0.3.10" \
  "docling>=2.0" \
  "paddleocr>=2.9,<3.0" \
  "ollama>=0.3"

# 3. PaddlePaddle (CPU or GPU — they cannot coexist in the same venv)
if $GPU_MODE; then
  echo "[3/5] Installing GPU packages (CUDA 12.4)..."
  pip install --quiet torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
  if pip install --quiet paddlepaddle-gpu \
       -i https://www.paddlepaddle.org.cn/packages/stable/cu123/; then
    echo "  paddlepaddle-gpu installed."
  else
    echo "  WARNING: paddlepaddle-gpu install failed — falling back to CPU build."
    pip install --quiet paddlepaddle
  fi
else
  echo "[3/5] Installing paddlepaddle (CPU)..."
  pip install --quiet paddlepaddle
fi

echo ""
echo "Optional: install Chandra (9B model, ~5 GB download):"
echo "  pip install 'chandra-ocr[hf]'"
echo ""

# 4. Pull glm-ocr model via Ollama
echo "[4/5] Pulling glm-ocr model via Ollama..."
if command -v ollama &>/dev/null; then
  if ! ollama list &>/dev/null 2>&1; then
    echo "  Starting ollama server..."
    ollama serve &>/dev/null &
    sleep 3
  fi
  ollama pull glm-ocr
  echo "  glm-ocr ready."
else
  echo "  WARNING: ollama not found on PATH. Make sure you're inside nix-shell."
fi

# 5. Web UI dependencies
echo "[5/5] Installing web UI dependencies..."
pip install --quiet \
  "fastapi>=0.111" \
  "uvicorn>=0.30" \
  "python-multipart>=0.0.9"

echo ""
echo "=== Setup complete ==="
echo "Usage:"
echo "  python ocr_pipeline.py <file.pdf> --engine all --compare"
echo "  python ocr_pipeline.py <file.pdf> --engine all --compare --gpu"
echo "  python ocr_pipeline.py <file.pdf> --engine cascade --gpu"
echo "  python ocr_pipeline.py <file.pdf> --engine tesseract --preprocess"
echo ""
echo "GPU setup (first time):"
echo "  ./setup.sh --gpu    # installs torch+cu124 and paddlepaddle-gpu"
echo ""
echo "Web UI:"
echo "  uvicorn app:app --reload --port 8000"
