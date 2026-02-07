# Python 3.12 Compatibility Fix

## Issue
Python 3.12 removed `pkgutil.ImpImporter`, causing issues with older setuptools versions.

## Quick Fix

If you're already in the virtual environment with the error:

```bash
# Activate your environment
source venv/bin/activate

# Upgrade setuptools (CRITICAL)
pip install --upgrade "setuptools>=65.5.1"

# Reinstall openmim
pip uninstall -y openmim
pip install -U openmim

# Now continue with MMDetection installation
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## Fresh Installation

If starting fresh:

```bash
# Remove old virtual environment
rm -rf venv

# Create new virtual environment
python3 -m venv venv
source venv/bin/activate

# CRITICAL: Upgrade pip and setuptools FIRST
pip install --upgrade pip
pip install --upgrade "setuptools>=65.5.1" wheel

# Now install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install openmim with fixed setuptools
pip install -U openmim

# Install MMDetection components
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet

# Install other dependencies
pip install -r requirements.txt
```

## Alternative: Use Python 3.11

If issues persist, use Python 3.11 instead:

```bash
# Install Python 3.11 (Ubuntu/Debian)
sudo apt update
sudo apt install python3.11 python3.11-venv

# Create venv with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Continue with normal setup
pip install --upgrade pip setuptools wheel
# ... rest of installation
```

## Verification

After fixing, verify:

```bash
# Check setuptools version (should be ≥65.5.1)
pip show setuptools

# Test mim
mim --version

# Test MMDetection
python -c "import mmdet; print(mmdet.__version__)"

# Test PyTorch with CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Updated setup.sh

I've created a fixed `setup_fixed.sh` that handles this automatically. Use that instead of the original `setup.sh`.

```bash
chmod +x setup_fixed.sh
./setup_fixed.sh
```
