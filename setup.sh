#!/bin/bash
set -e

echo "🚀 Starting Project Setup..."
# Cluster cant use on sudo or apt in finished container
# echo "System prerequisites (install swig)..."
# sudo apt update
# sudo apt install swig

echo "📦 Initializing submodules..."
git submodule update --init --recursive

echo "⚡ Installing dependencies with uv..."
if lspci | grep -iq "nvidia"; then
    echo "💚 NVIDIA GPU detected. Syncing with CUDA..."
    GPU_EXTRA="cuda"
elif lspci | grep -iq "amd"; then
    echo "❤️ AMD GPU detected. Syncing with ROCm..."
    GPU_EXTRA="rocm"
else
    echo "No supported GPU detected. Syncing base dependencies only."
    GPU_EXTRA="cpu"
fi

if [[ "$GPU_EXTRA" == "rocm" ]]; then
    cp pyproject.rocm.toml pyproject.toml
elif [[ "$GPU_EXTRA" == "cuda" || "$GPU_EXTRA" == "cpu" ]]; then
    cp pyproject.cuda.toml pyproject.toml
fi

uv sync

echo "✅ Done! Environment is ready."
