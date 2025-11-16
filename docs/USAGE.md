# Usage Guide

## Quick Start

### 1. Interactive Demo (5 minutes)

The fastest way to understand Mind Captioning:

```bash
# Activate environment
conda activate mcap_demo

# Launch Jupyter notebook
jupyter notebook mcap_demo.ipynb
```

The demo allows you to:
- Reconstruct arbitrary text from semantic features
- See the iterative optimization process
- Understand the text generation algorithm
- No heavy computation required

### 2. With Precomputed Results (1 hour)

Generate figures without running full analysis:

```bash
# Download precomputed results from figshare
# Extract to ./res/ directory

# Generate encoding figures
python mcap_summary_encoding.py

# Generate decoding figures
python mcap_summary_decoding.py
```

### 3. Full Analysis Pipeline (Weeks)

Run complete analysis from scratch:

```bash
# Step 1: Encoding (~1 day on multi-CPU)
python mcap_encoding_analysis.py

# Step 2: Decoding (~1-2 weeks on multi-CPU)
python mcap_decoding_analysis.py

# Step 3: Text Generation (~1 week on multi-GPU)
python mcap_analysis.py

# Step 4: Evaluation
python mcap_evaluation.py

# Step 5: Visualization
python mcap_summary_encoding.py
python mcap_summary_decoding.py
```

## Detailed Workflows

### Workflow A: Complete Analysis

For full reproducibility of manuscript results:

#### Step 1: Encoding Analysis

```bash
python mcap_encoding_analysis.py
```

**What it does**:
- Trains brain → features encoders
- Cross-validation on training data
- Generalization to test data
- Outputs encoding accuracies

**Time**: ~24 hours on 16-core CPU

**Outputs**: `./res/encoding/`

#### Step 2: Decoding Analysis

```bash
# Run without CV (faster)
python mcap_decoding_analysis.py

# Or with CV (recommended for validation)
python mcap_decoding_analysis.py --cv
```

**What it does**:
- Selects top voxels from encoding
- Trains brain → features decoders
- Decodes features from test brain activity

**Time**: ~1-2 weeks on 16-core CPU

**Outputs**: `./res/decoding/`

#### Step 3: Text Generation

```bash
python mcap_analysis.py
```

**What it does**:
- Loads decoded features
- Generates text descriptions
- Saves optimization trajectories

**Time**: ~1 week on 4× GPUs (16GB each)

**Outputs**: `./res/text_generation/`

#### Step 4: Evaluation

```bash
python mcap_evaluation.py
```

**What it does**:
- Computes similarity metrics
- BERT-Score, ROUGE, BLEU
- Identification accuracy

**Time**: ~1 day on GPU

**Outputs**: Updated results in `./res/text_generation/`

#### Step 5: Visualization

```bash
python mcap_summary_encoding.py
python mcap_summary_decoding.py
```

**What it does**:
- Generates all manuscript figures
- Statistical summaries
- Saves to `./fig/`

**Time**: ~30 minutes

### Workflow B: Faster Analysis (Demo Mode)

For quick testing or exploration:

#### Modify Configuration

Edit `util/mcap_config.py`:

```python
# Enable demo mode
do_demo = True  # Use fewer regularization parameters

# Reduce voxels
n_select_voxels = 5000  # Instead of 50000

# Process subset
sbjID = ['S1']  # Instead of all subjects
model_types = ['deberta-large']  # Single model only
```

#### Run Analysis

```bash
# Encoding (~2 hours)
python mcap_encoding_analysis.py

# Decoding (~1 day)
python mcap_decoding_analysis.py

# Text generation (~2 days on single GPU)
python mcap_analysis.py
```

**Note**: Results will differ slightly from manuscript but follow same trends.

### Workflow C: Skip to Text Generation

If you only care about text generation:

```bash
# Download pre-computed decoding results
# Extract decfeat_wb.zip to ./res/decoding/

# Run text generation directly
python mcap_analysis.py

# Evaluate
python mcap_evaluation.py

# Visualize
python mcap_summary_decoding.py
```

## Configuration

### Subject Selection

Process specific subjects:

```python
# In mcap_config.py
sbjID = ['S1', 'S2']  # Only subjects 1 and 2
```

Or modify script:

```python
# In analysis script
for sbj in ['S1', 'S3']:  # Custom selection
    # ... analysis code
```

### Model Selection

Choose language models:

```python
# In mcap_config.py
model_types = ['deberta-large']  # Only DeBERTa

# In mcap_analysis.py
LMType = 'deberta-large'  # Feature extraction
MLMType = 'roberta-large'  # Text generation
```

### ROI Selection

Specify brain regions:

```python
# In mcap_config.py
roi_types = ['WB', 'Lang']  # Whole brain and language regions
```

### GPU Configuration

```python
# In mcap_analysis.py
gpu_id = '0'  # Use GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# Multi-GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
```

### Regularization Parameters

```python
# In mcap_config.py
l2_param = {
    'nparam_log_search': 10,  # Number of lambdas to test
    'lowL': 1,                # 10^1
    'highL': 6                # 10^6
}

# For faster computation
l2_param = {
    'nparam_log_search': 4,   # Fewer parameters
    'lowL': 4,                # Smaller range
    'highL': 5
}
```

## Monitoring Progress

### Check Logs

```bash
# Encoding logs
tail -f ./res/encoding/trainPerception/deberta-large/S1/res_summary_log.txt

# Decoding logs
tail -f ./res/decoding/testPerception/deberta-large/S1/WB/layer10_log.txt

# Text generation logs
tail -f ./res/text_generation/testPerception/mlm_roberta-large/lm_deberta-large/S1/WB/log/log_samp0001_log.txt
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or
gpustat -i 1
```

### Check Results

```python
import scipy.io as sio

# Load encoding results
enc = sio.loadmat('./res/encoding/trainPerception/deberta-large/S1/layer10.mat')
print(f"Encoding accuracy: {enc['correlations'].mean():.4f}")

# Load decoding results
dec = sio.loadmat('./res/decoding/testPerception/deberta-large/S1/WB/layer10.mat')
print(f"Decoding accuracy: {dec['test_correlation']:.4f}")

# Load text generation results
txt = sio.loadmat('./res/text_generation/testPerception/mlm_roberta-large/lm_deberta-large/S1/WB/res/res_samp0001.mat')
print(f"Generated: {txt['best_cands'][-1]}")
print(f"Score: {txt['scores_all'][-1]:.4f}")
```

## Advanced Usage

### Parallel Processing

#### Multiple Machines

```bash
# Machine 1: Process S1, S2
export SUBJECTS="S1,S2"
python mcap_encoding_analysis.py

# Machine 2: Process S3, S4
export SUBJECTS="S3,S4"
python mcap_encoding_analysis.py
```

#### GNU Parallel

```bash
# Process all subjects in parallel
parallel -j 6 "python process_subject.py --subject {}" ::: S1 S2 S3 S4 S5 S6
```

### Custom Analysis

Create custom analysis script:

```python
from util.mcap_config import McapConfig
from util.mcap_encoding import cv_encoding, gen_encoding

# Custom configuration
config = McapConfig('./')
params = config.get_params()

# Modify parameters
params['sbjID'] = ['S1']
params['n_select_voxels'] = 10000

# Run analysis
cv_encoding(params)
gen_encoding(params)
```

### Batch Processing

Process multiple configurations:

```python
# batch_analysis.py
configs = [
    {'model': 'deberta-large', 'roi': 'WB'},
    {'model': 'deberta-large', 'roi': 'Lang'},
    # ... more configs
]

for cfg in configs:
    # Set configuration
    # Run analysis
    # Save results
```

## Common Tasks

### Task 1: Reproduce Figure 3 (Encoding)

```bash
# If you have encoding results
python mcap_summary_encoding.py

# If not, run encoding first
python mcap_encoding_analysis.py
python mcap_summary_encoding.py
```

### Task 2: Reproduce Figure 2, 4 (Decoding)

```bash
# If you have all results
python mcap_summary_decoding.py

# If not, run complete pipeline
python mcap_encoding_analysis.py
python mcap_decoding_analysis.py
python mcap_analysis.py
python mcap_evaluation.py
python mcap_summary_decoding.py
```

### Task 3: Generate Text for New Brain Data

```python
# Prepare new brain data
new_brain_data = ...  # Your fMRI data

# Load decoding model
dec_model = load_decoder('S1', 'deberta-large', 'WB')

# Decode features
decoded_features = dec_model.predict(new_brain_data)

# Generate text
from util.mcap_utils_demo import text_optimization_steps
generated_text = text_optimization_steps(decoded_features, ...)
```

### Task 4: Test New Language Model

```python
# In mcap_analysis.py
MLMType = 'your-new-model'  # e.g., 'gpt2-large'
LMType = 'your-feature-model'

# Run text generation
python mcap_analysis.py
```

## Output Interpretation

### Encoding Results

- **High correlation (>0.4)**: Strong encoding
- **Peak layer**: Best semantic representation
- **Voxel weights**: Feature importance

### Decoding Results

- **Correlation >0.3**: Good decoding
- **Decoded features**: Input for text generation
- **Identification accuracy**: Reliability metric

### Text Generation

- **Similarity score >0.5**: Good match
- **Generated text**: Quality varies by sample
- **Optimization trajectory**: Convergence behavior

## Troubleshooting

See [INSTALLATION.md](./INSTALLATION.md#troubleshooting) for common issues.

## Next Steps

- **Understand**: Read [OVERVIEW.md](./OVERVIEW.md)
- **Deep Dive**: See [API.md](./API.md)
- **Scripts**: Check [scripts/README.md](./scripts/README.md)
