#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Integrity Checker

This script checks if all required data files are present
and validates their structure.

Usage:
    python check_data.py
"""

import os
import sys


def check_data_integrity():
    """데이터 무결성 확인"""

    print("\n" + "=" * 60)
    print("MindCaptioning 데이터 무결성 확인")
    print("=" * 60 + "\n")

    all_good = True

    # 1. 캡션 데이터 (필수 - 이미 포함됨)
    print("[ 기본 데이터 ]")
    caption_file = 'data/caption/caption_ck20.csv'
    if os.path.exists(caption_file):
        try:
            import pandas as pd
            df = pd.read_csv(caption_file)
            print(f"  ✅ 캡션 데이터: {len(df):,} captions")
            if len(df) < 40000:
                print(f"     ⚠️  예상보다 적음 (예상: ~42,000)")
        except Exception as e:
            print(f"  ❌ 캡션 파일 읽기 오류: {e}")
            all_good = False
    else:
        print(f"  ❌ 캡션 파일 없음: {caption_file}")
        all_good = False

    # 2. fMRI 데이터 (전체 분석용)
    print("\n[ 전체 분석용 데이터 ]")
    fmri_dir = 'data/fmri/preprocessed'
    if os.path.exists(fmri_dir):
        fmri_files = [f for f in os.listdir(fmri_dir) if f.endswith('.mat')]
        print(f"  ✅ fMRI 데이터: {len(fmri_files)} files")

        # 예상 파일 수
        expected_files = [
            'trainPerception_S1.mat', 'trainPerception_S2.mat',
            'trainPerception_S3.mat', 'trainPerception_S4.mat',
            'trainPerception_S5.mat', 'trainPerception_S6.mat',
            'testPerception_S1.mat', 'testPerception_S2.mat',
            'testPerception_S3.mat', 'testPerception_S4.mat',
            'testPerception_S5.mat', 'testPerception_S6.mat',
            'testImagery_S1.mat', 'testImagery_S2.mat',
            'testImagery_S3.mat', 'testImagery_S4.mat',
            'testImagery_S5.mat', 'testImagery_S6.mat',
        ]

        # 파일 확인
        if fmri_files:
            try:
                import scipy.io as sio
                import numpy as np

                sample = os.path.join(fmri_dir, fmri_files[0])
                data = sio.loadmat(sample)
                print(f"     예시: {fmri_files[0]}")
                print(f"     - Shape: {data['dat'].shape}")
                print(f"     - Samples: {len(data['labels'])}")
                print(f"     - Voxels: {data['dat'].shape[1]:,}")

                # 데이터 품질 확인
                if np.isnan(data['dat']).any():
                    print(f"     ⚠️  NaN 값 발견!")
                if np.isinf(data['dat']).any():
                    print(f"     ⚠️  Inf 값 발견!")

            except Exception as e:
                print(f"     ❌ 파일 읽기 오류: {e}")

        missing = set(expected_files) - set(fmri_files)
        if missing:
            print(f"     ⚠️  누락된 파일 ({len(missing)}개):")
            for f in sorted(missing)[:5]:
                print(f"        - {f}")
            if len(missing) > 5:
                print(f"        ... and {len(missing) - 5} more")
    else:
        print(f"  ❌ fMRI 디렉토리 없음: {fmri_dir}")
        print(f"     → figshare에서 'preprocessed_fmri.zip' 다운로드 필요")
        all_good = False

    # 3. 의미 특징 (전체 분석용)
    feature_dir = 'data/feature/video/deberta-large'
    if os.path.exists(feature_dir):
        feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.mat')]
        print(f"  ✅ 의미 특징 (DeBERTa): {len(feature_files)} layers")

        if feature_files:
            try:
                import scipy.io as sio
                sample = os.path.join(feature_dir, feature_files[0])
                data = sio.loadmat(sample)
                print(f"     예시: {feature_files[0]}")
                print(f"     - Shape: {data['feat'].shape}")
                print(f"     - Videos: {data['feat'].shape[0]}")
                print(f"     - Features: {data['feat'].shape[1]}")
            except Exception as e:
                print(f"     ❌ 파일 읽기 오류: {e}")

        # 예상 레이어 수 확인
        if len(feature_files) < 20:
            print(f"     ⚠️  레이어 수가 적음 (예상: ~25 layers)")
    else:
        print(f"  ❌ 특징 디렉토리 없음: {feature_dir}")
        print(f"     → figshare에서 'features.zip' 다운로드 필요")
        all_good = False

    # 4. 사전 계산 결과 (선택)
    print("\n[ 선택적 데이터 (빠른 실행용) ]")
    result_dirs = [
        ('인코딩 결과', 'res/encoding/trainPerception/deberta-large', 'res_encoding.zip'),
        ('디코딩 결과', 'res/decoding/testPerception/deberta-large', 'decfeat_wb.zip'),
        ('텍스트 생성 결과', 'res/text_generation/testPerception', 'res_textgen.zip'),
    ]

    for name, path, zipfile in result_dirs:
        if os.path.exists(path):
            try:
                file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print(f"  ✅ {name}: {file_count} files")
            except:
                print(f"  ✅ {name}: 존재함")
        else:
            print(f"  ⚪ {name}: 없음")
            print(f"     → 필요시 figshare에서 '{zipfile}' 다운로드")

    # 5. 디스크 공간 확인
    print("\n[ 시스템 정보 ]")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        print(f"  디스크 공간:")
        print(f"    - 전체: {total / (1024**3):.1f} GB")
        print(f"    - 사용: {used / (1024**3):.1f} GB")
        print(f"    - 여유: {free / (1024**3):.1f} GB")

        if free < 50 * 1024**3:  # 50GB
            print(f"    ⚠️  여유 공간 부족 (최소 50GB 권장)")
    except Exception as e:
        print(f"  ⚠️  디스크 공간 확인 실패: {e}")

    # 최종 요약
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)

    if all_good and os.path.exists(fmri_dir) and os.path.exists(feature_dir):
        print("✅ 모든 필수 데이터가 준비되었습니다!")
        print("   → 전체 분석을 실행할 수 있습니다.")
        print("\n다음 단계:")
        print("  python mcap_encoding_analysis.py")
    elif os.path.exists('data/caption/caption_ck20.csv'):
        print("⚪ 기본 데이터만 있습니다.")
        print("   → 데모는 실행 가능합니다.")
        print("\n다음 단계:")
        print("  jupyter notebook mcap_demo.ipynb")
        print("\n전체 분석을 위해서는:")
        print("  1. figshare에서 데이터 다운로드")
        print("     https://doi.org/10.6084/m9.figshare.25808179")
        print("  2. preprocessed_fmri.zip 압축 해제")
        print("  3. features.zip 압축 해제")
    else:
        print("❌ 필수 데이터가 없습니다.")
        print("\n필요한 작업:")
        print("  1. 저장소 확인: git pull")
        print("  2. 데이터 다운로드 (상세: docs/DATA_PREPARATION.md)")

    print("=" * 60 + "\n")

    # 도움말
    print("더 많은 정보:")
    print("  - 데이터 준비: docs/DATA_PREPARATION.md")
    print("  - 빠른 시작: QUICK_START.md")
    print("  - 사용 가이드: docs/USAGE.md")
    print()


if __name__ == "__main__":
    try:
        check_data_integrity()
    except KeyboardInterrupt:
        print("\n\n중단됨.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
