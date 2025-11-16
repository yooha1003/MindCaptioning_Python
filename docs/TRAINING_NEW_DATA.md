# Training with New Data Guide

## Overview

이 가이드는 **자신의 fMRI 데이터**로 Mind Captioning 모델을 처음부터 훈련하는 방법을 설명합니다.

## Pre-trained Models 상황

### 이미 제공되는 것 (✅)
1. **Language Models** - Hugging Face에서 자동 다운로드
   - DeBERTa-large (의미 특징 추출)
   - RoBERTa-large (텍스트 생성)
   - BERT, GPT-2, T5 등

2. **Data** - figshare에서 다운로드 가능
   - 원본 논문의 전처리된 fMRI 데이터
   - 의미 특징 (semantic features)
   - 비디오 캡션

### 직접 학습해야 하는 것 (❌)
1. **Encoding Models** - Brain → Semantic Features
2. **Decoding Models** - Brain → Semantic Features

**이유**: 뇌 활동 패턴은 사람마다 다르므로, 각 피험자별로 개별 학습 필요

## 완전히 새로운 데이터로 훈련하기

### 필요한 것

#### 1. fMRI 데이터
```
요구사항:
- 피험자: 최소 1명 (권장: 3명 이상)
- 스캔: 3T fMRI scanner
- 훈련 데이터: 최소 500 samples (권장: 2000+)
- 테스트 데이터: 50-100 samples
- TR (Repetition Time): 2-3초
- 전처리: SPM, FSL, fMRIPrep 등으로 전처리 완료
```

#### 2. 자극 (Stimuli)
```
요구사항:
- 유형: 짧은 비디오 클립 (3초) 또는 이미지
- 개수: 훈련용 500+ / 테스트용 50+
- 캡션: 각 자극당 1개 이상의 텍스트 설명
```

#### 3. 컴퓨팅 자원
```
최소:
- CPU: 16+ cores
- RAM: 32GB+
- GPU: NVIDIA 16GB+ (텍스트 생성용)
- Storage: 200GB+

권장:
- CPU: 32+ cores
- RAM: 64GB+
- GPU: NVIDIA 24GB+ × 4
- Storage: 500GB+
```

## 단계별 훈련 프로세스

### Step 0: 데이터 준비

#### 0.1 fMRI 데이터 전처리

원시 fMRI 데이터를 전처리하여 MATLAB 형식으로 변환:

```python
# fmri_preprocessing.py
import numpy as np
import scipy.io as sio
import nibabel as nib

def preprocess_fmri_data(
    subject_id,
    fmri_files,      # List of fMRI nifti files
    labels,          # Stimulus labels for each volume
    runs,            # Run numbers
    output_path
):
    """
    전처리된 fMRI 데이터를 MindCaptioning 형식으로 변환

    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'S1')
    fmri_files : list
        List of preprocessed fMRI .nii files
    labels : ndarray
        Stimulus ID for each volume (1-indexed)
    runs : ndarray
        Run number for each volume
    output_path : str
        Output .mat file path
    """

    # Load and concatenate fMRI data
    all_data = []
    for fmri_file in fmri_files:
        img = nib.load(fmri_file)
        data = img.get_fdata()  # Shape: (x, y, z, time)

        # Reshape to (time, voxels)
        n_volumes = data.shape[3]
        data_2d = data.reshape(-1, n_volumes).T
        all_data.append(data_2d)

    # Concatenate across runs
    brain_data = np.vstack(all_data)  # Shape: (samples, voxels)

    # Remove NaN voxels
    valid_voxels = ~np.isnan(brain_data).any(axis=0)
    brain_data = brain_data[:, valid_voxels]

    # Z-score normalization per run
    for run_id in np.unique(runs):
        run_mask = runs == run_id
        brain_data[run_mask] = (
            brain_data[run_mask] - brain_data[run_mask].mean(axis=0)
        ) / brain_data[run_mask].std(axis=0)

    # Save in MindCaptioning format
    save_data = {
        'dat': brain_data,          # (samples, voxels)
        'labels': labels,           # (samples,) - stimulus IDs (1-indexed)
        'Run': runs,                # (samples,) - run numbers
        'Condition': np.ones_like(labels)  # All condition 1
    }

    sio.savemat(output_path, save_data)
    print(f"Saved: {output_path}")
    print(f"Shape: {brain_data.shape}")
    print(f"Samples: {len(labels)}, Voxels: {brain_data.shape[1]}")

# 사용 예시
preprocess_fmri_data(
    subject_id='S1',
    fmri_files=[
        'path/to/run1_preprocessed.nii',
        'path/to/run2_preprocessed.nii',
        # ... more runs
    ],
    labels=np.array([1, 2, 3, ..., 500]),  # Stimulus IDs
    runs=np.array([1, 1, 1, ..., 2, 2, ...]),  # Run numbers
    output_path='./data/fmri/preprocessed/trainPerception_S1.mat'
)
```

#### 0.2 캡션 데이터 준비

```python
# prepare_captions.py
import pandas as pd

# 캡션 데이터 생성
captions = []
for video_id in range(1, 501):  # 500 videos
    # 각 비디오에 대해 여러 캡션 작성 (권장: 5-20개)
    for i in range(5):
        caption = f"Your caption for video {video_id}, variant {i}"
        captions.append(caption)

# CSV 저장
df = pd.DataFrame({'caption': captions})
df.to_csv('./data/caption/caption_custom.csv', index=False)
```

#### 0.3 의미 특징 추출

```python
# extract_semantic_features.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import scipy.io as sio

def extract_semantic_features(
    captions,
    model_name='microsoft/deberta-large',
    output_dir='./data/feature/video/deberta-large/'
):
    """
    캡션에서 의미 특징 추출

    Parameters:
    -----------
    captions : list
        List of caption strings
    model_name : str
        Hugging Face model name
    output_dir : str
        Output directory for features
    """

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Extract features
    all_features = {}

    with torch.no_grad():
        for idx, caption in enumerate(captions):
            inputs = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # List of (batch, seq_len, hidden_size)

            # Store each layer's features
            for layer_idx, hidden_state in enumerate(hidden_states):
                # Average pooling over sequence
                features = hidden_state.mean(dim=1).cpu().numpy()  # (batch, hidden_size)

                layer_name = f'layer{layer_idx}'
                if layer_name not in all_features:
                    all_features[layer_name] = []
                all_features[layer_name].append(features)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(captions)} captions")

    # Save features
    import os
    os.makedirs(output_dir, exist_ok=True)

    for layer_name, features in all_features.items():
        features_array = np.vstack(features)  # (n_captions, hidden_size)

        # Average over multiple captions per video
        n_videos = len(captions) // 5  # Assuming 5 captions per video
        video_features = []
        for i in range(n_videos):
            video_feat = features_array[i*5:(i+1)*5].mean(axis=0)
            video_features.append(video_feat)

        video_features = np.vstack(video_features)

        save_path = f"{output_dir}/{layer_name}.mat"
        sio.savemat(save_path, {'feat': video_features})
        print(f"Saved: {save_path}, Shape: {video_features.shape}")

# 사용 예시
captions = pd.read_csv('./data/caption/caption_custom.csv')['caption'].tolist()
extract_semantic_features(captions)
```

### Step 1: Encoding 모델 학습

```bash
# 준비된 데이터로 encoding 학습
python mcap_encoding_analysis.py
```

**설정 조정** (`util/mcap_config.py`):

```python
# 자신의 피험자 ID로 변경
sbjID = ['MySubject1', 'MySubject2']  # Your subject IDs

# 자신의 모델 타입으로 변경
model_types = ['deberta-large']  # Or your model

# 데모 모드 (빠른 테스트)
do_demo = True  # Set False for final training
```

**예상 시간**:
- Demo mode: ~2-4 hours
- Full mode: ~1 day (16-core CPU)

**출력**: `./res/encoding/trainPerception/{model_type}/{subject}/`

### Step 2: Decoding 모델 학습

```bash
# Encoding 결과를 사용하여 decoding 학습
python mcap_decoding_analysis.py
```

**예상 시간**:
- Demo mode: ~1 day
- Full mode: ~1-2 weeks (16-core CPU)

**출력**: `./res/decoding/testPerception/{model_type}/{subject}/{roi}/`

### Step 3: 텍스트 생성

```bash
# Decoded features로부터 텍스트 생성
python mcap_analysis.py
```

**설정 조정** (`mcap_analysis.py`):

```python
# 자신의 캡션 파일 사용
capType = 'custom'  # Your caption file name (caption_custom.csv)

# 피험자 설정
sbjs = ['MySubject1', 'MySubject2']

# GPU 설정
gpu_id = '0'  # Your GPU ID
```

**예상 시간**:
- Single GPU: ~30 days per ROI
- 4× GPUs: ~1 week

**출력**: `./res/text_generation/testPerception/...`

### Step 4: 평가

```bash
python mcap_evaluation.py
```

**출력**: 평가 메트릭 (BERT-Score, ROUGE, BLEU 등)

### Step 5: 시각화

```bash
python mcap_summary_encoding.py
python mcap_summary_decoding.py
```

**출력**: `./fig/encoding/`, `./fig/decoding/`

## 실전 예제: 나만의 데이터셋

### 예제 시나리오
- 피험자 1명
- 100개 비디오로 훈련
- 20개 비디오로 테스트

```python
# 1. 데이터 구조 생성
import os
os.makedirs('./data/fmri/preprocessed/', exist_ok=True)
os.makedirs('./data/caption/', exist_ok=True)
os.makedirs('./data/feature/video/deberta-large/', exist_ok=True)

# 2. fMRI 전처리 (위의 코드 사용)
preprocess_fmri_data(
    subject_id='MySubject',
    fmri_files=[...],  # Your preprocessed fMRI files
    labels=np.repeat(np.arange(1, 101), 20),  # 100 videos × 20 repetitions
    runs=np.tile(np.arange(1, 11), 200),  # 10 runs
    output_path='./data/fmri/preprocessed/trainPerception_MySubject.mat'
)

# Test data
preprocess_fmri_data(
    subject_id='MySubject',
    fmri_files=[...],  # Test fMRI files
    labels=np.repeat(np.arange(1, 21), 4),  # 20 videos × 4 repetitions
    runs=np.tile(np.arange(1, 3), 40),  # 2 runs
    output_path='./data/fmri/preprocessed/testPerception_MySubject.mat'
)

# 3. 캡션 준비
captions = [
    "Person walking in park",
    "Dog running on beach",
    # ... 100 videos × 5 captions each
]

# 4. 의미 특징 추출
extract_semantic_features(captions)

# 5. Config 수정
# Edit util/mcap_config.py:
# sbjID = ['MySubject']

# 6. 학습 실행
# python mcap_encoding_analysis.py
# python mcap_decoding_analysis.py
# python mcap_analysis.py
```

## 데이터 최소 요구사항

### 신뢰할 수 있는 결과를 위한 최소값

| 항목 | 최소값 | 권장값 | 이상적 |
|------|--------|--------|--------|
| 훈련 비디오/이미지 | 100 | 500 | 2000+ |
| 테스트 비디오/이미지 | 20 | 50 | 100 |
| 비디오당 반복 | 10 | 20 | 30+ |
| 비디오당 캡션 | 1 | 5 | 20 |
| 피험자 수 | 1 | 3 | 6+ |
| fMRI 스캔 시간 | 2시간 | 4시간 | 8시간 |

### 데이터 품질 체크리스트

```python
# data_quality_check.py
import scipy.io as sio
import numpy as np

def check_fmri_data_quality(fmri_path):
    """fMRI 데이터 품질 확인"""
    data = sio.loadmat(fmri_path)

    brain_data = data['dat']
    labels = data['labels'].flatten()
    runs = data['Run'].flatten()

    print(f"\n=== Data Quality Report ===")
    print(f"Shape: {brain_data.shape}")
    print(f"Samples: {len(labels)}")
    print(f"Voxels: {brain_data.shape[1]}")
    print(f"Runs: {len(np.unique(runs))}")
    print(f"Unique stimuli: {len(np.unique(labels))}")
    print(f"Repetitions per stimulus: {len(labels) / len(np.unique(labels)):.1f}")

    # Check for NaN/Inf
    nan_count = np.isnan(brain_data).sum()
    inf_count = np.isinf(brain_data).sum()
    print(f"\nData integrity:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")

    # Check signal quality
    snr = brain_data.mean() / brain_data.std()
    print(f"\nSignal quality:")
    print(f"  Mean: {brain_data.mean():.4f}")
    print(f"  Std: {brain_data.std():.4f}")
    print(f"  SNR: {snr:.4f}")

    # Recommendations
    print(f"\n=== Recommendations ===")
    if brain_data.shape[0] < 500:
        print("⚠️  Low sample count. Consider collecting more data.")
    else:
        print("✅ Sample count looks good")

    if brain_data.shape[1] < 10000:
        print("⚠️  Few voxels. Check preprocessing.")
    else:
        print("✅ Voxel count looks good")

    if nan_count > 0 or inf_count > 0:
        print("❌ Data contains NaN/Inf. Clean data first!")
    else:
        print("✅ Data integrity OK")

# 사용
check_fmri_data_quality('./data/fmri/preprocessed/trainPerception_MySubject.mat')
```

## 트러블슈팅

### 문제 1: 학습이 너무 오래 걸림

**해결책**:
```python
# util/mcap_config.py에서 조정
do_demo = True  # 파라미터 수 줄이기
n_select_voxels = 5000  # 기본 50000에서 줄이기
```

### 문제 2: Encoding accuracy가 너무 낮음 (<0.1)

**원인**:
- 데이터 품질 문제
- 전처리 오류
- 데이터 양 부족

**해결책**:
1. 데이터 품질 확인
2. 더 많은 데이터 수집
3. 전처리 파라미터 조정

### 문제 3: 텍스트 생성이 의미없음

**원인**:
- Decoding accuracy가 낮음
- 캡션 품질 문제

**해결책**:
1. Decoding 모델 재학습
2. 더 나은 캡션 작성
3. 더 많은 반복 데이터

## Pre-trained Model 활용 (Transfer Learning)

원본 논문의 학습된 가중치를 시작점으로 사용:

```python
# transfer_learning.py
import scipy.io as sio
import numpy as np

# 원본 모델 로드
original_weights = sio.loadmat(
    './res/encoding/trainPerception/deberta-large/S1/layer10.mat'
)

# Fine-tune on your data
# (Encoding/Decoding 학습 시 initial weights로 사용)
```

**주의**: 뇌 패턴이 사람마다 다르므로 transfer learning 효과가 제한적일 수 있음

## 결론

완전히 새로운 데이터로 훈련하려면:

1. ✅ **Language Models**: 자동 다운로드 (걱정 없음)
2. ❌ **Brain Models**: 직접 학습 필요
3. 📊 **데이터**: 최소 100 videos × 10 repetitions
4. ⏱️ **시간**: Demo ~1주일, Full ~1개월
5. 💻 **자원**: Multi-CPU + Multi-GPU 권장

**가장 중요한 것**: 고품질의 fMRI 데이터와 충분한 양의 반복 측정!
