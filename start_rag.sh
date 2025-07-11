#!/bin/bash

# Enhanced RAG System Startup Script - GPU Mode
# Cancel HSA Override, restore MinerU GPU mode

echo "🚀 Starting RAG System - GPU Mode"
echo "================================"

# Check conda environment
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    exit 1
fi

# Activate conda environment
echo "🔄 Activating conda environment: rag_system"
source ~/miniconda3/bin/activate ryzen_ai_ta

if [ $? -ne 0 ]; then
    echo "❌ Error: Unable to activate conda environment ryzen_ai_ta"
    exit 1
fi

# Set environment variables to optimize ROCm performance
echo "🔧 Configuring ROCm environment variables"
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0

# Optimize ROCm memory management (cancel HSA Override)
export HSA_XNACK=1
export AMD_SERIALIZE_KERNEL=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set GPU optimization parameters
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="gfx1100;gfx1101;gfx1102;gfx1151"

echo "🔍 Environment variables check:"
echo "  ROCM_PATH: $ROCM_PATH"
echo "  HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "  HSA_XNACK: $HSA_XNACK"
echo "  AMD_SERIALIZE_KERNEL: $AMD_SERIALIZE_KERNEL"
echo "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

# Check GPU status
echo "🔍 Checking GPU status:"
if command -v rocm-smi &> /dev/null; then
    echo "📊 ROCm GPU information:"
    rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true
else
    echo "⚠️ rocm-smi not found, skipping GPU check"
fi

# Check necessary services
echo "🔍 Checking Ollama service:"
if pgrep -f "ollama serve" > /dev/null; then
    echo "✅ Ollama service is running"
else
    echo "⚠️ Ollama service is not running, please start Ollama first"
fi

# Check Python packages
echo "🔍 Checking key Python packages:"
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"
python -c "import chromadb; print(f'✅ ChromaDB: {chromadb.__version__}')"

# Create necessary directories
echo "📁 Creating necessary directories"
mkdir -p chroma_db uploaded_docs/mineru_output

# Clear old GPU cache
echo "🧹 Clearing GPU cache"
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU cache cleared')"

# Set signal handling
cleanup() {
    echo "🛑 Shutting down RAG system..."
    echo "🧹 Cleaning up GPU resources..."
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    echo "✅ Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "================================"
echo "🚀 Starting RAG Application (GPU Mode)"
echo "📡 Access URL: http://localhost:7861"
echo "🔧 MinerU GPU mode enabled"
echo "⚠️ Press Ctrl+C to safely shutdown"
echo "================================"

# Start RAG application
python rag_app.py 
