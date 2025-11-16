# 데이터 준비 가이드

## 빠른 체크리스트

### ✅ 바로 실행 가능 (데이터 다운로드 없음)
- [ ] `mcap_demo.ipynb` - 인터랙티브 데모

### 📥 데이터 다운로드 필요

#### 전체 분석용
- [ ] `preprocessed_fmri.zip` (30GB) - fMRI 데이터
- [ ] `features.zip` (5GB) - 의미 특징

#### 그림 생성만 (빠름)
- [ ] `res_encoding.zip` (2GB) - 인코딩 결과
- [ ] `res_textgen.zip` (5GB) - 텍스트 생성 결과
- [ ] `decfeat_wb.zip` (3GB, 선택) - 디코딩 결과

---

## 상세 다운로드 가이드

### 1. figshare 접속

**URL**: https://doi.org/10.6084/m9.figshare.25808179

### 2. 파일 다운로드

#### A. 웹 브라우저 사용
1. figshare 링크 방문
2. 필요한 파일 클릭
3. "Download" 버튼 클릭

#### B. 명령줄 사용 (Linux/Mac)

```bash
# figshare에서 파일 ID 확인 후 다운로드
# 예시 (실제 URL은 figshare에서 확인)

# 전처리된 fMRI 데이터
wget -O preprocessed_fmri.zip "https://figshare.com/ndownloader/files/XXXXX"

# 의미 특징
wget -O features.zip "https://figshare.com/ndownloader/files/XXXXX"

# 사전 계산 결과
wget -O res_encoding.zip "https://figshare.com/ndownloader/files/46420405"
wget -O res_textgen.zip "https://figshare.com/ndownloader/files/46422523"
wget -O decfeat_wb.zip "https://figshare.com/ndownloader/files/46420294"
```

### 3. 압축 해제

```bash
# MindCaptioning_Python 디렉토리에서 실행

# fMRI 데이터
unzip preprocessed_fmri.zip -d ./data/fmri/preprocessed/

# 의미 특징
unzip features.zip -d ./data/feature/

# 인코딩 결과
unzip res_encoding.zip -d ./res/encoding/

# 텍스트 생성 결과
unzip res_textgen.zip -d ./res/text_generation/

# 디코딩 결과
unzip decfeat_wb.zip -d ./res/decoding/
```

### 4. 구조 확인

```bash
# 올바른 구조 확인
python -c "
import os

dirs = [
    'data/fmri/preprocessed',
    'data/feature/video/deberta-large',
    'data/caption',
    'res/encoding',
    'res/decoding',
    'res/text_generation'
]

print('=== 디렉토리 확인 ===')
for d in dirs:
    exists = '✅' if os.path.exists(d) else '❌'
    count = len(os.listdir(d)) if os.path.exists(d) else 0
    print(f'{exists} {d} ({count} files)')
"
```

---

## 파일 상세 정보

### 1. fMRI 데이터 (preprocessed_fmri.zip)

**크기**: ~30GB
**형식**: MATLAB .mat 파일

**포함 파일**:
```
trainPerception_S1.mat  # 피험자 1, 훈련 데이터
trainPerception_S2.mat  # 피험자 2, 훈련 데이터
...
trainPerception_S6.mat  # 피험자 6, 훈련 데이터

testPerception_S1.mat   # 피험자 1, 테스트 (지각)
testPerception_S2.mat
...
testPerception_S6.mat

testImagery_S1.mat      # 피험자 1, 테스트 (심상)
testImagery_S2.mat
...
testImagery_S6.mat
```

**파일 구조**:
```python
import scipy.io as sio

data = sio.loadmat('trainPerception_S1.mat')
# data['dat']       : (samples, voxels) - 뇌 활동 데이터
# data['labels']    : (samples, 1) - 자극 ID (1-indexed)
# data['Run']       : (samples, 1) - 런 번호
# data['Condition'] : (samples, 1) - 조건 (모두 1)
```

**필요한 경우**:
- Encoding 분석 실행
- Decoding 분석 실행

---

### 2. 의미 특징 (features.zip)

**크기**: ~5GB
**형식**: MATLAB .mat 파일

**디렉토리 구조**:
```
video/
├── deberta-large/
│   ├── layer0.mat    # DeBERTa layer 0 features
│   ├── layer1.mat
│   ├── ...
│   └── layer24.mat   # DeBERTa layer 24
└── timesformer/
    ├── layer0.mat
    └── ...
```

**파일 구조**:
```python
data = sio.loadmat('deberta-large/layer10.mat')
# data['feat'] : (n_videos, feature_dim) - 의미 특징
# 예: (2108, 1024) - 2108 videos × 1024 dimensions
```

**필요한 경우**:
- Encoding 분석 실행
- Decoding 분석 실행

---

### 3. 캡션 데이터 (이미 포함됨!)

**위치**: `data/caption/caption_ck20.csv`
**크기**: ~500KB
**형식**: CSV

**구조**:
```csv
caption
"A person walking in a park"
"A person walking in a park on a sunny day"
...
```

- 2108개 비디오
- 각 비디오당 20개 캡션
- 총 42,160개 캡션

---

### 4. 사전 계산 결과

#### A. res_encoding.zip (2GB)

**포함 내용**:
```
encoding/
├── trainPerception/
│   ├── deberta-large/
│   │   ├── S1/
│   │   │   ├── layer0.mat
│   │   │   ├── layer1.mat
│   │   │   └── ...
│   │   ├── S2/
│   │   └── ...
│   └── timesformer/
│       └── ...
└── testPerception/
    └── ...
```

**용도**: 그림 3 (인코딩 정확도) 생성

#### B. res_textgen.zip (5GB)

**포함 내용**:
```
text_generation/
├── testPerception/
│   └── mlm_roberta-large/
│       └── lm_deberta-large/
│           ├── S1/
│           │   ├── WB/
│           │   │   ├── res/
│           │   │   │   ├── res_samp0001.mat
│           │   │   │   ├── res_samp0002.mat
│           │   │   │   └── ...
│           │   │   └── log/
│           │   └── Lang/
│           ├── S2/
│           └── ...
└── testImagery/
    └── ...
```

**용도**: 그림 2, 4 (텍스트 생성 결과) 생성

#### C. decfeat_wb.zip (3GB)

**포함 내용**:
```
decoding/
├── testPerception/
│   └── deberta-large/
│       ├── S1/
│       │   └── WB/
│       │       ├── layer0.mat
│       │       └── ...
│       └── ...
└── testImagery/
    └── ...
```

**용도**: 텍스트 생성 입력 (디코딩 건너뛰기)

---

## 디스크 공간 요구사항

| 시나리오 | 필요 공간 |
|----------|-----------|
| 데모만 | < 1GB |
| 그림 생성 | ~10GB |
| 전체 분석 | ~50GB |
| 모든 데이터 | ~100GB |

---

## 데이터 검증 스크립트

```python
# check_data.py
import os
import scipy.io as sio
import numpy as np

def check_data_integrity():
    """데이터 무결성 확인"""

    print("=" * 60)
    print("데이터 무결성 확인")
    print("=" * 60)

    # 1. 캡션 데이터
    caption_file = 'data/caption/caption_ck20.csv'
    if os.path.exists(caption_file):
        import pandas as pd
        df = pd.read_csv(caption_file)
        print(f"✅ 캡션: {len(df)} lines")
    else:
        print(f"❌ 캡션 파일 없음: {caption_file}")

    # 2. fMRI 데이터
    fmri_dir = 'data/fmri/preprocessed'
    if os.path.exists(fmri_dir):
        fmri_files = [f for f in os.listdir(fmri_dir) if f.endswith('.mat')]
        print(f"✅ fMRI 데이터: {len(fmri_files)} files")

        # 샘플 파일 확인
        if fmri_files:
            sample = os.path.join(fmri_dir, fmri_files[0])
            data = sio.loadmat(sample)
            print(f"   예시: {fmri_files[0]}")
            print(f"   - Shape: {data['dat'].shape}")
            print(f"   - Samples: {len(data['labels'])}")
            print(f"   - Voxels: {data['dat'].shape[1]}")
    else:
        print(f"❌ fMRI 디렉토리 없음: {fmri_dir}")

    # 3. 의미 특징
    feature_dir = 'data/feature/video/deberta-large'
    if os.path.exists(feature_dir):
        feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.mat')]
        print(f"✅ 의미 특징: {len(feature_files)} layers")

        # 샘플 확인
        if feature_files:
            sample = os.path.join(feature_dir, feature_files[0])
            data = sio.loadmat(sample)
            print(f"   예시: {feature_files[0]}")
            print(f"   - Shape: {data['feat'].shape}")
    else:
        print(f"❌ 특징 디렉토리 없음: {feature_dir}")

    # 4. 결과 (선택)
    result_dirs = [
        ('인코딩 결과', 'res/encoding/trainPerception/deberta-large'),
        ('디코딩 결과', 'res/decoding/testPerception/deberta-large'),
        ('텍스트 생성 결과', 'res/text_generation/testPerception')
    ]

    print("\n선택적 데이터:")
    for name, path in result_dirs:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"⚪ {name}: 없음 (필요시 다운로드)")

    print("=" * 60)

if __name__ == "__main__":
    check_data_integrity()
```

**실행**:
```bash
python check_data.py
```

---

## 문제 해결

### Q1: 다운로드가 너무 느림
**A**: 브라우저 대신 `wget` 또는 `curl` 사용 권장

### Q2: 압축 해제 실패
**A**:
```bash
# 디스크 공간 확인
df -h .

# 충분한 공간 확보 후 재시도
unzip -t file.zip  # 파일 무결성 확인
```

### Q3: 파일이 어디에 있는지 모르겠음
**A**:
```bash
find . -name "*.mat" -type f | head -10
```

### Q4: figshare 다운로드 링크를 찾을 수 없음
**A**:
1. https://doi.org/10.6084/m9.figshare.25808179 방문
2. "Files" 섹션에서 파일 클릭
3. "Download" 버튼 오른쪽 클릭 → "링크 주소 복사"

---

## 다음 단계

데이터 준비 완료 후:

1. **데모 실행**:
   ```bash
   jupyter notebook mcap_demo.ipynb
   ```

2. **전체 분석**:
   ```bash
   python mcap_encoding_analysis.py
   python mcap_decoding_analysis.py
   python mcap_analysis.py
   ```

3. **그림 생성**:
   ```bash
   python mcap_summary_encoding.py
   python mcap_summary_decoding.py
   ```

더 많은 정보: [QUICK_START.md](../QUICK_START.md)
