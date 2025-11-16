# Quick Start Guide

## 🚀 3가지 사용 시나리오

### 시나리오 1: 논문 결과 재현 (기존 데이터)

```bash
# 1. 설치
git clone https://github.com/yooha1003/MindCaptioning_Python.git
cd MindCaptioning_Python
bash setup.sh
conda activate mcap_demo

# 2. 데이터 다운로드 (figshare)
# https://doi.org/10.6084/m9.figshare.25808179

# 3. 빠른 데모
jupyter notebook mcap_demo.ipynb

# 4. 전체 분석 (데이터 다운로드 후)
python mcap_encoding_analysis.py
python mcap_decoding_analysis.py
python mcap_analysis.py
python mcap_evaluation.py
```

### 시나리오 2: 사전 계산된 결과로 그림 생성

```bash
# 1. 설치 (위와 동일)

# 2. 결과 다운로드 (figshare)
# - res_encoding.zip
# - res_textgen.zip
# 압축 해제: ./res/ 디렉토리

# 3. 그림 생성
python mcap_summary_encoding.py
python mcap_summary_decoding.py
```

### 시나리오 3: **새로운 데이터로 훈련** ⭐

```bash
# 1. 설치 (위와 동일)

# 2. 데이터 준비
# a) fMRI 전처리
python prepare_new_data.py --mode fmri \
    --subject MySubject \
    --fmri-files run1.nii run2.nii run3.nii \
    --labels-file labels.npy \
    --runs-file runs.npy \
    --output ./data/fmri/preprocessed/trainPerception_MySubject.mat

# b) 의미 특징 추출
python prepare_new_data.py --mode features \
    --captions-file my_captions.csv \
    --output ./data/feature/video/deberta-large/

# c) 데이터 품질 확인
python prepare_new_data.py --mode check \
    --check-file ./data/fmri/preprocessed/trainPerception_MySubject.mat

# 3. Config 수정
# Edit util/mcap_config.py:
# sbjID = ['MySubject']

# 4. 학습 실행
python mcap_encoding_analysis.py   # ~1 day
python mcap_decoding_analysis.py   # ~1-2 weeks
python mcap_analysis.py            # ~1 week

# 5. 평가 및 시각화
python mcap_evaluation.py
python mcap_summary_encoding.py
python mcap_summary_decoding.py
```

## 📊 Pre-trained Models

### 있는 것 (✅ 자동 다운로드)
- **Language Models**: DeBERTa, RoBERTa, BERT 등
- Hugging Face에서 자동으로 다운로드됨
- 별도 설치 불필요

### 없는 것 (❌ 직접 학습)
- **Brain Encoding Models**: Brain → Features
- **Brain Decoding Models**: Brain → Features
- 각 피험자마다 개별 학습 필요
- 이유: 뇌 패턴이 사람마다 다름

## 🔧 새 데이터 준비 상세

### 필요한 것

#### 1. fMRI 데이터
```python
# 형식: 전처리된 NIfTI files (.nii or .nii.gz)
# 최소: 100 videos × 10 repetitions = 1000 samples
# 권장: 500 videos × 20 repetitions = 10000 samples
```

#### 2. Labels & Runs
```python
import numpy as np

# Labels: 각 볼륨이 어떤 자극인지 (1-indexed)
labels = np.array([1, 1, 1, ..., 2, 2, 2, ...])  # Shape: (n_volumes,)
np.save('labels.npy', labels)

# Runs: 각 볼륨이 어떤 런인지
runs = np.array([1, 1, 1, ..., 2, 2, 2, ...])  # Shape: (n_volumes,)
np.save('runs.npy', runs)
```

#### 3. Captions
```python
import pandas as pd

# CSV 파일: 각 행이 하나의 캡션
captions = [
    "A person walking in the park",
    "A person walking in the park on a sunny day",  # Same video, different caption
    "Walking person outdoors",                      # Same video, different caption
    "Dog running on beach",
    "Dog playing in ocean water",
    # ... 최소 1개, 권장 5-20개 per video
]

df = pd.DataFrame({'caption': captions})
df.to_csv('my_captions.csv', index=False)
```

### 데이터 준비 예제

```python
# example_prepare_data.py
import numpy as np
import pandas as pd
import nibabel as nib

# 1. 실험 정보
n_videos = 100
n_reps_per_video = 10
n_runs = 10
volumes_per_run = 100

# 2. Labels 생성 (각 비디오를 10번 반복)
labels = np.repeat(np.arange(1, n_videos + 1), n_reps_per_video)
np.save('labels.npy', labels)

# 3. Runs 생성
runs = np.tile(np.arange(1, n_runs + 1), volumes_per_run)
np.save('runs.npy', runs)

# 4. fMRI 파일 리스트
fmri_files = [
    'path/to/sub-01_run-01_bold_preprocessed.nii.gz',
    'path/to/sub-01_run-02_bold_preprocessed.nii.gz',
    # ... 10 runs
]

# 5. Captions 작성
captions = []
for video_id in range(1, n_videos + 1):
    for cap_variant in range(5):  # 5 captions per video
        caption = f"Description of video {video_id}, variant {cap_variant}"
        captions.append(caption)

df = pd.DataFrame({'caption': captions})
df.to_csv('my_captions.csv', index=False)

# 6. 전처리 실행
# python prepare_new_data.py --mode fmri ...
```

## ⏱️ 예상 소요 시간

| 단계 | Demo Mode | Full Mode | 비고 |
|------|-----------|-----------|------|
| Encoding | 2-4 hours | 1 day | 16-core CPU |
| Decoding | 1 day | 1-2 weeks | 16-core CPU |
| Text Gen | 2 days | 1 week | 4× GPU (16GB) |
| **Total** | **3-4 days** | **2-3 weeks** | Parallel processing |

## 💾 디스크 용량

| 데이터 | 용량 |
|--------|------|
| fMRI (원본) | ~50GB per subject |
| fMRI (전처리) | ~10GB per subject |
| Features | ~5GB |
| Results | ~20GB |
| **Total** | **~100GB** |

## 🔍 데이터 품질 확인

```bash
# 데이터 준비 후 품질 확인
python prepare_new_data.py --mode check \
    --check-file ./data/fmri/preprocessed/trainPerception_MySubject.mat
```

**기대 출력:**
```
Data Quality Report
===================
Data shape: (1000, 65000)
  Samples: 1000
  Voxels: 65000
  Runs: 10
  Unique stimuli: 100
  Repetitions/stimulus: 10.0

✅ Sample count looks good
✅ Voxel count looks good
✅ Data integrity OK
✅ Repetition count looks good
```

## 📖 더 많은 정보

- **새 데이터 훈련 가이드**: [docs/TRAINING_NEW_DATA.md](./docs/TRAINING_NEW_DATA.md)
- **설치 가이드**: [docs/INSTALLATION.md](./docs/INSTALLATION.md)
- **사용 가이드**: [docs/USAGE.md](./docs/USAGE.md)
- **프로젝트 개요**: [docs/OVERVIEW.md](./docs/OVERVIEW.md)

## ❓ FAQ

### Q1: Pre-trained brain model은 어디서 받나요?
**A**: Brain model은 제공되지 않습니다. 각 사람의 뇌 패턴이 다르므로 자신의 데이터로 직접 학습해야 합니다.

### Q2: 최소 몇 개의 비디오가 필요한가요?
**A**: 최소 100개 (권장 500개), 각 비디오당 최소 10번 반복 (권장 20번)

### Q3: fMRI 전처리는 어떻게 하나요?
**A**: SPM, FSL, fMRIPrep 등 표준 도구 사용. 전처리 완료된 NIfTI 파일이 필요합니다.

### Q4: GPU 없이 가능한가요?
**A**: Encoding/Decoding은 가능 (CPU만). Text generation은 GPU 강력 권장 (30일 → 1주일).

### Q5: 한 명의 피험자만으로도 되나요?
**A**: 네, 가능합니다. 하지만 일반화를 위해 3명 이상 권장.

## 🆘 도움말

문제가 발생하면:
1. [docs/INSTALLATION.md](./docs/INSTALLATION.md#troubleshooting) - 설치 문제
2. [docs/TRAINING_NEW_DATA.md](./docs/TRAINING_NEW_DATA.md) - 새 데이터 훈련
3. GitHub Issues - 버그 리포트
