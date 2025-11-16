# mcap_encoding_analysis.py

## Overview

Performs encoding analysis to evaluate how well semantic features extracted by language models can be predicted from brain activity patterns.

## Purpose

Encoding analysis answers the question: "Can we predict semantic features from brain activity?" This is a necessary first step before decoding, as it:

1. Validates that semantic information is present in brain activity
2. Identifies which voxels encode semantic information
3. Determines optimal layers for feature representation
4. Provides weights for later voxel selection

## Usage

### Basic Usage

```bash
python mcap_encoding_analysis.py
```

### From Python

```python
from mcap_encoding_analysis import EncodingAnalysis

# Initialize analysis
analysis = EncodingAnalysis(root_path='./')

# Run cross-validation encoding
analysis.run_cv_encoding()

# Run generalization encoding
analysis.run_gen_encoding()

# Or run both
analysis.run_all()
```

## Workflow

### 1. Initialization
- Loads configuration parameters
- Sets up directories
- Initializes logging

### 2. Cross-Validation Encoding
- **Data**: Training perception data
- **Method**: Nested cross-validation
- **Purpose**: Parameter selection and validation
- **Output**: CV accuracy, optimal regularization

### 3. Generalization Encoding
- **Data**: Train on perception, test on perception/imagery
- **Method**: Train on all training data, test on held-out
- **Purpose**: Generalization performance
- **Output**: Test accuracy, encoding weights

## Parameters

Located in `util/mcap_config.py`:

```python
# Regularization parameters
l2_param = {
    'nparam_log_search': 10,  # Number of lambda values to test
    'lowL': 1,                # Lower bound (10^1)
    'highL': 6                # Upper bound (10^6)
}

# Demo mode (faster)
do_demo = False  # Set True for 4 params from 10^4 to 10^5
```

## Inputs

### Required Data

1. **fMRI Data**: `./data/fmri/preprocessed/`
   - Format: MATLAB `.mat` files
   - Structure:
     - `dat`: Brain activity (samples × voxels)
     - `Run`: Run numbers for each sample
     - `Condition`: Condition labels
     - `labels`: Stimulus IDs

2. **Semantic Features**: `./data/feature/video/{model_type}/`
   - Format: MATLAB `.mat` files per layer
   - Structure:
     - `feat`: Features (stimuli × dimensions)

### Configuration

- `model_types`: Language models to use (e.g., 'deberta-large')
- `sbjID`: Subject IDs (S1-S6)
- `data_type`: 'trainPerception', 'testPerception', 'testImagery'

## Outputs

### Cross-Validation Results
**Location**: `./res/encoding/trainPerception/{model_type}/{subject}/`

**Files**:
- `layer*.mat`: Results for each layer
- `res_summary.mat`: Aggregated results

**Contents**:
- `correlations`: CV accuracy (folds × voxels)
- `best_lambda`: Optimal regularization parameter
- `predictions`: Predicted features

### Generalization Results
**Location**: `./res/encoding/{test_type}/{model_type}/{subject}/`

**Files**: Same structure as CV results

**Contents**:
- `test_correlation`: Test set accuracy
- `weights`: Regression weights (voxels × features)
- `predictions`: Predicted features on test set

## Algorithm Details

### Nested Cross-Validation

```
For each fold in outer CV:
    Hold out test fold

    For each lambda in regularization parameters:
        For each fold in inner CV:
            Train on inner train
            Test on inner test
            Compute correlation
        Average across inner folds

    Select best lambda
    Train on all outer train with best lambda
    Test on outer test fold
```

### Generalization

```
For each lambda in regularization parameters:
    For each fold in training CV:
        Train on train fold
        Test on validation fold
        Compute correlation
    Average across folds

Select best lambda
Train on all training data with best lambda
Test on held-out test data
```

### Ridge Regression

```
W = (X^T X + λI)^(-1) X^T Y

Where:
- X: Brain activity (samples × voxels)
- Y: Semantic features (samples × dimensions)
- λ: Regularization parameter
- W: Weights (voxels × dimensions)
```

## Performance Considerations

### Memory Requirements
- Scales with: n_voxels × n_features × n_samples
- Typical: ~10GB RAM per subject
- Can reduce by processing layers sequentially

### Computation Time
- **Single CPU**: ~1 week total
- **Multi-CPU (16 cores)**: ~1 day
- Parallelizable across:
  - Subjects
  - Model types
  - Layers

### Optimization Tips

1. **Reduce parameter search space**:
   ```python
   l2_param['nparam_log_search'] = 4  # Instead of 10
   ```

2. **Demo mode**:
   ```python
   do_demo = True  # Faster but less accurate
   ```

3. **Process subset**:
   - Select specific subjects
   - Select specific models
   - Select specific layers

## Expected Results

### Accuracy
- **Mean correlation**: 0.2 - 0.5 depending on layer
- **Peak layers**: Mid-to-late transformer layers (8-12)
- **Subject variability**: ±0.1 correlation

### Best Layers by Model

| Model | Best Layer | Mean r |
|-------|-----------|--------|
| DeBERTa-large | 10-12 | 0.45 |
| BERT-base | 6-8 | 0.35 |
| GPT-2 | 8-10 | 0.30 |

*(Approximate values, vary by subject)*

## Troubleshooting

### Memory Errors
```python
# Reduce memory usage
- Process fewer layers at once
- Reduce n_voxels by ROI selection
- Use float32 instead of float64
```

### Slow Performance
```python
# Speed up computation
- Enable parallel processing
- Reduce parameter search space
- Use demo mode
- Process on HPC cluster
```

### Missing Data
```
Error: Cannot find fMRI data for subject X

Solution:
- Download from figshare
- Check data paths in config
- Verify file structure
```

## Validation

### Check Results

```python
import scipy.io as sio

# Load results
res = sio.loadmat('./res/encoding/trainPerception/deberta-large/S1/layer10.mat')

# Check accuracy
print("CV correlations:", res['correlations'].shape)
print("Mean accuracy:", res['correlations'].mean())

# Check weights
print("Weights shape:", res['weights'].shape)
```

### Expected Output

```
CV correlations: (6, 70000)  # 6 folds, 70k voxels
Mean accuracy: 0.42
Weights shape: (70000, 1024)  # 70k voxels, 1024 features
```

## Related Scripts

- `mcap_decoding_analysis.py`: Uses encoding results for voxel selection
- `mcap_summary_encoding.py`: Visualizes encoding results
- `util/mcap_encoding.py`: Core encoding functions
- `util/mcap_config.py`: Configuration parameters

## References

- Original MATLAB: `code/matlab/mcap_encoding_analysis.m`
- Algorithm: Ridge regression with nested CV
- Similar to: fMRI encoding models (Naselaris et al., 2011)

## Author

Python port by Claude (2025)
Based on original work by Tomoyasu Horikawa
