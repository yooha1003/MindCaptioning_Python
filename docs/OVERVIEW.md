# MindCaptioning Python - Project Overview

## Introduction

MindCaptioning is a groundbreaking neuroscience project that generates natural language descriptions of visual mental imagery directly from human brain activity recorded via fMRI scans. This Python implementation provides a complete, unified codebase converting the original MATLAB/Python hybrid system into pure Python.

## Scientific Background

### The Challenge

How can we decode and articulate what people are thinking? This project addresses this fundamental question in cognitive neuroscience by:

1. Recording brain activity while subjects view video clips
2. Learning mappings between brain activity and semantic features
3. Decoding semantic features from new brain activity
4. Generating natural language descriptions from decoded features

### The Approach

The Mind Captioning system uses a three-stage pipeline:

```
fMRI Brain Activity → Semantic Features → Natural Language Text
     (Encoding/Decoding)      (Text Generation)
```

## System Architecture

### 1. Encoding Analysis

**Purpose**: Learn how semantic features are represented in brain activity

**Process**:
- Train regression models: Brain Activity → Semantic Features
- Evaluate prediction accuracy using cross-validation
- Identify best layers and voxels for encoding

**Key Components**:
- `mcap_encoding_analysis.py`: Main encoding script
- `util/mcap_encoding.py`: Encoding algorithms
- L2-regularized linear regression
- Multi-layer feature analysis

**Outputs**:
- Encoding accuracy per layer and subject
- Voxel weights for each semantic dimension
- Optimal layer selection

### 2. Decoding Analysis

**Purpose**: Predict semantic features from brain activity

**Process**:
- Select informative voxels based on encoding results
- Train decoders: Brain Activity → Semantic Features
- Decode features from test brain activity
- Evaluate decoding accuracy

**Key Components**:
- `mcap_decoding_analysis.py`: Main decoding script
- `util/mcap_decoding.py`: Decoding algorithms
- ROI-based voxel selection
- Ridge regression for stability

**Outputs**:
- Decoded semantic features
- Decoding accuracy metrics
- Voxel importance scores

### 3. Text Generation

**Purpose**: Generate natural language from decoded features

**Process**:
- Start with initial token (e.g., [UNK])
- Iteratively refine text using Masked Language Model
- Optimize alignment with decoded brain features
- Select best candidates via beam search

**Key Components**:
- `mcap_analysis.py`: Text generation script
- `mcap_evaluation.py`: Evaluation metrics
- Masked Language Models (RoBERTa, BERT, etc.)
- Semantic feature extraction (DeBERTa, etc.)

**Outputs**:
- Generated text descriptions
- Similarity scores with reference captions
- Optimization trajectories

## Data Flow

```
Input Data:
├── fMRI Brain Activity (Voxels × Time)
├── Video Stimuli IDs
└── Reference Captions

Encoding Stage:
├── Extract Semantic Features from Captions
├── Train: Brain → Features
└── Output: Encoding Models, Accuracies

Decoding Stage:
├── Select Top Voxels
├── Train: Brain → Features
├── Decode Test Brain Activity
└── Output: Decoded Features

Text Generation:
├── Initialize Text
├── Mask & Predict Tokens
├── Align with Decoded Features
├── Optimize & Select Best
└── Output: Generated Descriptions

Evaluation:
├── Compare Generated vs Reference
├── Compute Similarity Metrics
├── Visualization
└── Output: Figures, Statistics
```

## Key Technologies

### Deep Learning Models

1. **Masked Language Models (MLM)**:
   - RoBERTa-large (default)
   - BERT variants
   - DeBERTa

2. **Feature Extraction Models**:
   - DeBERTa-large (default)
   - BERT, GPT-2, T5
   - CLIP, Sentence-BERT

### Machine Learning Methods

1. **Regression**:
   - Ridge regression (L2 regularization)
   - Cross-validation for parameter selection
   - Nested CV for unbiased evaluation

2. **Optimization**:
   - Beam search
   - Greedy decoding
   - Sampling strategies

3. **Evaluation Metrics**:
   - Pearson correlation
   - BERT-Score
   - ROUGE
   - BLEU
   - Identification accuracy

## Experimental Design

### Subjects
- 6 participants (S1-S6)
- Healthy adults with normal vision

### Data Acquisition
- 3T fMRI scanner
- Training: ~2100 samples per subject
- Testing: 72 perception + 72 imagery samples

### Stimuli
- Short video clips (3s)
- Diverse content (emotions, actions, scenes)
- 20 reference captions per video

### ROI Analysis
- **WB**: Whole Brain
- **Lang**: Language regions
- **WBnoLang**: Whole brain excluding language
- **WBnoSem**: Whole brain excluding semantic
- **WBnoVis**: Whole brain excluding visual

## Performance

### Encoding
- Peak accuracy: r ≈ 0.4-0.5
- Best layers: Mid-to-late transformer layers
- Varies by subject and model

### Decoding
- Correlation with true features: r ≈ 0.3-0.4
- Whole brain generally outperforms ROIs
- Consistent across subjects

### Text Generation
- Identification accuracy: ~40% (top-1/100)
- High semantic similarity with references
- Meaningful descriptions even with lower accuracy

## Computational Requirements

### Hardware
- **CPU**: Multi-core recommended (16+ cores)
- **GPU**: CUDA-capable (16GB+ VRAM for text gen)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for full dataset

### Time Estimates

| Stage | Single CPU/GPU | Multi CPU/GPU | With Precomputed |
|-------|---------------|---------------|------------------|
| Encoding | ~1 week | ~1 day | N/A |
| Decoding | ~50 weeks | ~1-2 weeks | N/A |
| Text Gen | ~30 days/ROI | ~1 week | N/A |
| Total | ~2+ years | ~1-2 months | ~1 week |

**Recommendation**: Download precomputed results from figshare for figure generation

## Python Implementation Advantages

### Over Original MATLAB/Python Hybrid

1. **Unified Environment**:
   - Single language, no MATLAB license needed
   - Easier setup and dependency management

2. **Open Source**:
   - All code uses open-source libraries
   - Community-driven improvements

3. **Cross-Platform**:
   - Works on Linux, macOS, Windows
   - Containerization support (Docker)

4. **Modern Practices**:
   - Object-oriented design
   - Type hints and documentation
   - Better error handling

5. **Integration**:
   - Easy integration with Python ML ecosystem
   - Jupyter notebook support
   - Hugging Face compatibility

## Research Applications

### Current
- Understanding neural representation of semantics
- Brain-computer interfaces
- Cognitive neuroscience research

### Future Potential
- Assistive technology for communication disorders
- Dream/imagery decoding
- Lie detection and memory probing
- AI-guided neuroscience discovery

## Ethical Considerations

This technology raises important ethical questions:

1. **Privacy**: Mental privacy and thought decoding
2. **Consent**: When and how to use such technology
3. **Applications**: Appropriate vs inappropriate uses
4. **Accuracy**: Limitations and potential misuse
5. **Regulation**: Need for ethical guidelines

**Current State**: Technology requires willing participation, explicit consent, and extended fMRI scanning. It cannot "read minds" in everyday settings.

## References

### Primary Publication
Horikawa, T. (2025). Mind captioning: Evolving descriptive text of mental content from human brain activity. *Science Advances*, 11(45).

### Related Work
- fMRI data preprocessing: SPM, FSL
- Deep language models: Hugging Face Transformers
- Video stimuli: Cowen & Keltner (2017)

## Getting Started

1. **Quick Demo**: `jupyter notebook mcap_demo.ipynb`
2. **Full Pipeline**: See [USAGE.md](./USAGE.md)
3. **API Reference**: See [API.md](./API.md)
4. **Installation**: See [INSTALLATION.md](./INSTALLATION.md)

## Support

- **Documentation**: [./docs/](.)
- **Issues**: GitHub Issues
- **Data**: [figshare](https://doi.org/10.6084/m9.figshare.25808179)
- **Paper**: [Science Advances](https://doi.org/10.1126/sciadv.adw1464)
