#!/bin/bash

set -e  # Exit on error

echo "================================================"
echo "Vehicle Lights Pipeline - Setup Script"
echo "================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script requires Linux${NC}"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. NVIDIA GPU required.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
nvidia-smi --query-gpu=name --format=csv,noheader

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; }; then
    echo -e "${RED}Error: Python 3.9+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Create virtual environment
echo ""
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo ""
echo -e "${YELLOW}Installing PyTorch with CUDA 11.8...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MMDetection dependencies
echo ""
echo -e "${YELLOW}Installing MMCV and MMDetection...${NC}"
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet

# Install other dependencies
echo ""
echo -e "${YELLOW}Installing other dependencies...${NC}"
pip install -r requirements.txt

# Create directory structure
echo ""
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p data/vehicle_lights/{train,val,test,annotations}
mkdir -p models
mkdir -p outputs/{logs,videos}
mkdir -p checkpoints

# Download pre-trained RTMDet-m model (if not exists)
if [ ! -f "models/rtmdet_m_8xb32-300e_coco.pth" ]; then
    echo ""
    echo -e "${YELLOW}Downloading RTMDet-m pretrained weights...${NC}"
    wget -P models/ https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
    mv models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth models/rtmdet_m_8xb32-300e_coco.pth
fi

# Run tests
echo ""
echo -e "${YELLOW}Running tests...${NC}"
pytest tests/ -v

# Verify GPU access
echo ""
echo -e "${YELLOW}Verifying GPU access...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python main.py --config configs/default.yaml"
echo ""
echo "To run with overlay visualization:"
echo "  python main.py --config configs/default.yaml --overlay live"
echo ""
