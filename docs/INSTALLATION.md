# Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (12.0+), or Windows 10+
- **Python**: 3.7 or higher
- **RAM**: 16GB
- **Storage**: 50GB free space
- **CPU**: 4+ cores

### Recommended Requirements
- **OS**: Linux (Ubuntu 22.04+)
- **Python**: 3.8+
- **RAM**: 32GB+
- **Storage**: 100GB+ free space
- **CPU**: 16+ cores
- **GPU**: NVIDIA GPU with 16GB+ VRAM (for text generation)
- **CUDA**: 11.0+ (if using GPU)

## Installation Methods

### Method 1: Conda (Recommended)

#### Step 1: Install Conda
If you don't have Conda installed:
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### Step 2: Clone Repository
```bash
git clone https://github.com/yooha1003/MindCaptioning_Python.git
cd MindCaptioning_Python
```

#### Step 3: Create Environment
```bash
# Run setup script
bash setup.sh

# Activate environment
conda activate mcap_demo
```

#### Step 4: Verify Installation
```bash
python -c "import torch; import transformers; print('Installation successful!')"
```

### Method 2: pip + venv

#### Step 1: Clone Repository
```bash
git clone https://github.com/yooha1003/MindCaptioning_Python.git
cd MindCaptioning_Python
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Verify Installation
```bash
python -c "import torch; import transformers; print('Installation successful!')"
```

### Method 3: Docker (Coming Soon)

```bash
# Pull Docker image
docker pull mindcaptioning/python:latest

# Run container
docker run -it --gpus all mindcaptioning/python:latest
```

## Data Setup

### Download Data

1. **Visit figshare**: https://doi.org/10.6084/m9.figshare.25808179

2. **Download files**:
   - `preprocessed_fmri.zip` (Required for encoding/decoding)
   - `features.zip` (Required for encoding/decoding)
   - `decfeat_wb.zip` (Optional: Skip decoding, use for text gen)
   - `res_encoding.zip` (Optional: Skip encoding, for figures)
   - `res_textgen.zip` (Optional: Skip text gen, for figures)

3. **Download captions**: Already included in `./data/caption/`

### Extract Data

```bash
cd MindCaptioning_Python

# Extract preprocessed fMRI data
unzip preprocessed_fmri.zip -d ./data/fmri/preprocessed/

# Extract features
unzip features.zip -d ./data/feature/

# (Optional) Extract precomputed results
unzip decfeat_wb.zip -d ./res/decoding/
unzip res_encoding.zip -d ./res/encoding/
unzip res_textgen.zip -d ./res/text_generation/
```

### Verify Data Structure

```bash
tree -L 3 -d data/
```

Expected output:
```
data/
├── caption
│   └── caption_ck20.csv
├── feature
│   └── video
│       ├── deberta-large
│       └── timesformer
└── fmri
    └── preprocessed
        ├── trainPerception_S1.mat
        ├── trainPerception_S2.mat
        └── ...
```

## GPU Setup (Optional but Recommended)

### Check GPU Availability

```bash
# Check NVIDIA GPU
nvidia-smi

# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

### Install CUDA Toolkit

If you don't have CUDA installed:

```bash
# Ubuntu
sudo apt-get install nvidia-cuda-toolkit

# Or download from NVIDIA website
# https://developer.nvidia.com/cuda-downloads
```

### Configure GPU

In analysis scripts, set GPU device:

```python
# Use GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Use multiple GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
```

## Testing Installation

### Quick Test

```bash
# Activate environment
conda activate mcap_demo

# Run quick test
python -c "
import numpy as np
import scipy.io
import torch
import transformers
from util.mcap_config import McapConfig
print('All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers version: {transformers.__version__}')
"
```

### Demo Test

```bash
# Run demo notebook
jupyter notebook mcap_demo.ipynb
```

## Troubleshooting

### Issue: CUDA not available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Install/update NVIDIA drivers
2. Install CUDA toolkit
3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
   ```

### Issue: Out of memory

**Symptom**: `CUDA out of memory` or system freezes

**Solutions**:
1. Reduce batch size in config
2. Use smaller model
3. Process fewer samples at once
4. Use CPU instead of GPU (slower)

### Issue: Import errors

**Symptom**: `ModuleNotFoundError`

**Solutions**:
1. Activate correct environment:
   ```bash
   conda activate mcap_demo
   ```
2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Issue: Data not found

**Symptom**: `FileNotFoundError` when running scripts

**Solutions**:
1. Verify data paths:
   ```bash
   ls -lh data/fmri/preprocessed/
   ls -lh data/feature/video/
   ```
2. Re-extract downloaded files
3. Check `mcap_config.py` paths

### Issue: Permission denied

**Symptom**: Cannot write to directories

**Solutions**:
```bash
# Fix permissions
chmod -R u+w ./res/
chmod -R u+w ./fig/

# Or run with appropriate permissions
```

## Next Steps

After successful installation:

1. **Quick Start**: Run demo notebook
   ```bash
   jupyter notebook mcap_demo.ipynb
   ```

2. **Full Analysis**: See [USAGE.md](./USAGE.md)

3. **Understanding**: Read [OVERVIEW.md](./OVERVIEW.md)

4. **API Details**: See [API.md](./API.md)

## Support

- **GitHub Issues**: https://github.com/yooha1003/MindCaptioning_Python/issues
- **Documentation**: ./docs/
- **Original Paper**: https://doi.org/10.1126/sciadv.adw1464
