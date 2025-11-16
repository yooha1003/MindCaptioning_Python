# MindCaptioning (Full Python Implementation)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-red.svg)

## Overview

This is a **complete Python implementation** of the Mind Captioning project, converting the original MATLAB/Python hybrid codebase to pure Python.

**Original Work:**
> Horikawa, T. (2025) Mind captioning: Evolving descriptive text of mental content from human brain activity. *Science Advances*, 11(45).

This project demonstrates a novel approach to generate natural language descriptions of visual mental imagery directly from human brain activity recorded via fMRI scans.

## What's New in This Python Port

✅ **Full Python Implementation**: All MATLAB encoding/decoding analyses converted to Python
✅ **Unified Codebase**: Single language implementation for easier setup and maintenance
✅ **Improved Documentation**: Comprehensive docs for all scripts and functions
✅ **Modern Python Practices**: Object-oriented design, type hints, and better error handling
✅ **Enhanced Visualization**: matplotlib/seaborn-based plotting for all figures
✅ **Cross-platform**: Works on Linux, macOS, and Windows

## Repository Structure

```
MindCaptioning_Python/
├── mcap_analysis.py              # Text generation from decoded features
├── mcap_evaluation.py            # Evaluation of generated text
├── mcap_encoding_analysis.py     # NEW: Encoding analysis (MATLAB→Python)
├── mcap_decoding_analysis.py     # NEW: Decoding analysis (MATLAB→Python)
├── mcap_summary_encoding.py      # NEW: Encoding results visualization
├── mcap_summary_decoding.py      # NEW: Decoding results visualization
├── mcap_demo.ipynb               # Interactive demo notebook
│
├── util/                         # Utility modules
│   ├── mcap_config.py            # Configuration management
│   ├── mcap_encoding.py          # Encoding functions
│   ├── mcap_decoding.py          # Decoding functions
│   ├── mcap_utils.py             # General utilities
│   ├── mcap_utils_demo.py        # Demo utilities
│   └── thutil4.py                # Helper functions
│
├── data/                         # Data directory
│   ├── fmri/                     # fMRI data
│   ├── feature/                  # Semantic features
│   ├── caption/                  # Video captions
│   └── model/                    # Pre-trained models
│
├── res/                          # Results directory
│   ├── encoding/                 # Encoding results
│   ├── decoding/                 # Decoding results
│   └── text_generation/          # Generated text
│
├── fig/                          # Figures directory
│   ├── encoding/                 # Encoding figures
│   └── decoding/                 # Decoding figures
│
├── docs/                         # Documentation
│   ├── OVERVIEW.md               # Project overview
│   ├── INSTALLATION.md           # Installation guide
│   ├── USAGE.md                  # Usage guide
│   ├── API.md                    # API reference
│   └── scripts/                  # Script documentation
│
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── setup.sh                      # Setup script
└── README.md                     # This file
```

## Key Features

### 🧠 Encoding Analysis
- Cross-validation encoding to evaluate feature prediction from brain activity
- Generalization encoding for train/test evaluation
- Multi-layer analysis for deep language models
- Voxel selection based on encoding accuracy

### 🔍 Decoding Analysis
- Cross-validation decoding for validation
- Generalization decoding for test data
- ROI-based analysis (Whole Brain, Language regions, etc.)
- Feature decoding for text generation

### 📝 Text Generation
- Masked Language Model (MLM) based generation
- Iterative optimization with beam search
- Multiple sampling strategies
- Semantic feature alignment

### 📊 Visualization & Evaluation
- Encoding accuracy plots by layer and subject
- Decoding performance visualization
- Generated text quality metrics
- Comparison with reference captions

## Installation

### Prerequisites
- Python 3.7 or higher
- CUDA-capable GPU (recommended for text generation)
- 16GB+ RAM (32GB+ recommended)
- 100GB+ free disk space (for data)

### Option 1: Conda Environment (Recommended)

```bash
# Clone repository
git clone https://github.com/yooha1003/MindCaptioning_Python.git
cd MindCaptioning_Python

# Create and activate conda environment
bash setup.sh
conda activate mcap_demo
```

### Option 2: pip Install

```bash
# Clone repository
git clone https://github.com/yooha1003/MindCaptioning_Python.git
cd MindCaptioning_Python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

### Download Datasets

1. **Raw fMRI Data**: Available on [OpenNeuro](https://openneuro.org/datasets/ds005191)
   - DOI: 10.18112/openneuro.ds005191.v1.0.2

2. **Preprocessed Data**: Available on [figshare](https://doi.org/10.6084/m9.figshare.25808179)
   - Preprocessed fMRI: `preprocessed_fmri.zip`
   - Semantic features: `features.zip`
   - Decoding results: `decfeat_wb.zip`
   - Encoding results: `res_encoding.zip`
   - Text generation results: `res_textgen.zip`

### Data Placement

```bash
# Place downloaded data in appropriate directories
./data/fmri/preprocessed/          # Preprocessed fMRI data
./data/feature/video/               # Semantic features
./data/caption/                     # Video captions
./res/encoding/                     # Encoding results (optional)
./res/decoding/                     # Decoding results (optional)
./res/text_generation/              # Text gen results (optional)
```

## Usage

### Quick Demo

```bash
# Activate environment
conda activate mcap_demo

# Run interactive demo
jupyter notebook mcap_demo.ipynb
```

### Complete Analysis Pipeline

#### 1. Encoding Analysis
```bash
# Run encoding analysis (generates semantic feature predictions)
python mcap_encoding_analysis.py

# Visualize encoding results
python mcap_summary_encoding.py
```

**Time Estimate**: ~1 week on multi-CPU system

#### 2. Decoding Analysis
```bash
# Run decoding analysis (decodes features from brain activity)
python mcap_decoding_analysis.py

# Optional: Include cross-validation (time-consuming)
python mcap_decoding_analysis.py --cv

# Visualize decoding results
python mcap_summary_decoding.py
```

**Time Estimate**: ~50 weeks on single CPU, ~1-2 weeks on multi-CPU system

#### 3. Text Generation
```bash
# Generate text from decoded features
python mcap_analysis.py

# Evaluate generated text
python mcap_evaluation.py
```

**Time Estimate**: ~30 days per ROI on single GPU, ~1 week on multi-GPU system

## Configuration

### Adjusting Parameters

Edit parameters in `util/mcap_config.py`:

```python
# Regularization parameters
l2_param['nparam_log_search'] = 10  # Reduce for faster computation
l2_param['lowL'] = 1
l2_param['highL'] = 6

# Voxel selection
n_select_voxels = 50000  # Reduce (e.g., 5000) for faster computation

# Demo mode (faster but less accurate)
do_demo = True  # Set False for manuscript-quality results
```

### GPU Configuration

Set GPU device in analysis scripts:

```python
gpu_id = '0'  # Use GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# For multi-GPU:
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
```

## Results

### Expected Outputs

- **Encoding Results**: `./res/encoding/`
  - Feature prediction accuracy by layer
  - Voxel weights for each semantic dimension

- **Decoding Results**: `./res/decoding/`
  - Decoded semantic features from brain activity
  - Decoding accuracy metrics

- **Text Generation**: `./res/text_generation/`
  - Generated text descriptions
  - Similarity scores with reference captions
  - Optimization trajectories

- **Figures**: `./fig/`
  - Encoding accuracy plots (Fig. 3)
  - Decoding performance (Fig. 2, 4)
  - Text quality visualizations

### Performance Metrics

From the original paper:
- **Decoding Accuracy**: ~0.3-0.4 correlation with reference features
- **Text Identification**: ~40% accuracy (top-1 from 100 candidates)
- **Semantic Similarity**: High correlation between generated and reference texts

## Documentation

Detailed documentation is available in the `./docs/` folder:

- [Installation Guide](./docs/INSTALLATION.md)
- [Usage Guide](./docs/USAGE.md)
- [API Reference](./docs/API.md)
- [Script Documentation](./docs/scripts/)

## Differences from Original

### Converted from MATLAB to Python:
- `mcap_encoding_analysis.m` → `mcap_encoding_analysis.py`
- `mcap_decoding_analysis.m` → `mcap_decoding_analysis.py`
- `mcap_summary_encoding.m` → `mcap_summary_encoding.py`
- `mcap_summary_decoding.m` → `mcap_summary_decoding.py`
- All utility functions in `util/`

### Implementation Changes:
- Ridge regression via `scikit-learn` instead of custom MATLAB implementation
- matplotlib/seaborn for plotting instead of MATLAB figures
- Object-oriented design for better modularity
- Improved error handling and logging

### Preserved:
- All analysis algorithms and mathematical procedures
- Parameter values from the manuscript
- Text generation pipeline
- Evaluation metrics

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{horikawa2025mind,
  title={Mind captioning: Evolving descriptive text of mental content from human brain activity},
  author={Horikawa, Tomoyasu},
  journal={Science Advances},
  volume={11},
  number={45},
  year={2025},
  publisher={American Association for the Advancement of Science}
}
```

## License

This project inherits the license from the original MindCaptioning repository.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
- Open an issue on GitHub
- Check the [documentation](./docs/)
- Contact: [maintainer email]

## Acknowledgments

- Original MATLAB implementation: Tomoyasu Horikawa
- Python port: Claude (2025)
- Original research: NTT Communication Science Laboratories

## Changelog

### Version 1.0.0 (2025-11-16)
- Initial release of full Python implementation
- Complete MATLAB to Python conversion
- Added comprehensive documentation
- Improved code organization and modularity

---

**Note**: This is a research code. For production use, additional optimization and testing are recommended.
