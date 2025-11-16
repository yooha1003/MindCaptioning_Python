#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper Script: Prepare New Data for Training

This script helps you prepare your own fMRI data and stimuli
for training MindCaptioning models.

Usage:
    python prepare_new_data.py --help
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import pandas as pd
import nibabel as nib
from pathlib import Path


def preprocess_fmri_data(subject_id, fmri_files, labels, runs, output_path):
    """
    Convert preprocessed fMRI data to MindCaptioning format

    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'S1')
    fmri_files : list
        List of preprocessed fMRI .nii/.nii.gz files
    labels : ndarray
        Stimulus ID for each volume (1-indexed)
    runs : ndarray
        Run number for each volume
    output_path : str
        Output .mat file path
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing fMRI data for {subject_id}")
    print(f"{'='*60}")

    # Load and concatenate fMRI data
    all_data = []
    for i, fmri_file in enumerate(fmri_files):
        print(f"Loading file {i+1}/{len(fmri_files)}: {fmri_file}")
        img = nib.load(fmri_file)
        data = img.get_fdata()  # Shape: (x, y, z, time)

        # Reshape to (time, voxels)
        n_volumes = data.shape[3]
        data_2d = data.reshape(-1, n_volumes).T
        all_data.append(data_2d)
        print(f"  Shape: {data.shape}, Volumes: {n_volumes}")

    # Concatenate across runs
    brain_data = np.vstack(all_data)  # Shape: (samples, voxels)
    print(f"\nConcatenated shape: {brain_data.shape}")

    # Remove NaN voxels
    valid_voxels = ~np.isnan(brain_data).any(axis=0)
    n_removed = (~valid_voxels).sum()
    brain_data = brain_data[:, valid_voxels]
    print(f"Removed {n_removed} voxels with NaN values")
    print(f"Remaining voxels: {brain_data.shape[1]}")

    # Z-score normalization per run
    print("\nNormalizing data per run...")
    for run_id in np.unique(runs):
        run_mask = runs == run_id
        run_data = brain_data[run_mask]
        brain_data[run_mask] = (run_data - run_data.mean(axis=0)) / (run_data.std(axis=0) + 1e-8)
        print(f"  Run {run_id}: {run_mask.sum()} volumes")

    # Verify data
    assert brain_data.shape[0] == len(labels), "Mismatch in samples and labels"
    assert brain_data.shape[0] == len(runs), "Mismatch in samples and runs"

    # Save in MindCaptioning format
    save_data = {
        'dat': brain_data,              # (samples, voxels)
        'labels': labels.reshape(-1, 1),  # (samples, 1) - stimulus IDs (1-indexed)
        'Run': runs.reshape(-1, 1),       # (samples, 1) - run numbers
        'Condition': np.ones((len(labels), 1))  # All condition 1
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sio.savemat(output_path, save_data)

    print(f"\n{'='*60}")
    print(f"✅ Saved: {output_path}")
    print(f"   Samples: {len(labels)}")
    print(f"   Voxels: {brain_data.shape[1]}")
    print(f"   Unique stimuli: {len(np.unique(labels))}")
    print(f"   Repetitions/stimulus: {len(labels) / len(np.unique(labels)):.1f}")
    print(f"{'='*60}\n")


def extract_semantic_features(captions, model_name='microsoft/deberta-large',
                              output_dir='./data/feature/video/deberta-large/',
                              n_captions_per_video=5):
    """
    Extract semantic features from captions using a language model

    Parameters:
    -----------
    captions : list
        List of caption strings
    model_name : str
        Hugging Face model name
    output_dir : str
        Output directory for features
    n_captions_per_video : int
        Number of captions per video (for averaging)
    """
    print(f"\n{'='*60}")
    print(f"Extracting semantic features")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    import torch
    from transformers import AutoTokenizer, AutoModel

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")

    # Extract features
    all_features = {}

    print(f"\nProcessing {len(captions)} captions...")
    with torch.no_grad():
        for idx, caption in enumerate(captions):
            inputs = tokenizer(caption, return_tensors='pt', padding=True,
                             truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)

            # Store each layer's features
            for layer_idx, hidden_state in enumerate(hidden_states):
                # Average pooling over sequence
                features = hidden_state.mean(dim=1).cpu().numpy()  # (1, hidden_size)

                layer_name = f'layer{layer_idx}'
                if layer_name not in all_features:
                    all_features[layer_name] = []
                all_features[layer_name].append(features)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(captions)} captions")

    # Save features
    os.makedirs(output_dir, exist_ok=True)

    n_videos = len(captions) // n_captions_per_video
    print(f"\nAveraging {n_captions_per_video} captions per video...")
    print(f"Total videos: {n_videos}")

    for layer_name, features in all_features.items():
        features_array = np.vstack(features)  # (n_captions, hidden_size)

        # Average over multiple captions per video
        video_features = []
        for i in range(n_videos):
            start_idx = i * n_captions_per_video
            end_idx = (i + 1) * n_captions_per_video
            video_feat = features_array[start_idx:end_idx].mean(axis=0)
            video_features.append(video_feat)

        video_features = np.vstack(video_features)

        save_path = f"{output_dir}/{layer_name}.mat"
        sio.savemat(save_path, {'feat': video_features})
        print(f"  Saved {layer_name}: {save_path}, Shape: {video_features.shape}")

    print(f"\n✅ Feature extraction completed!")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*60}\n")


def check_data_quality(fmri_path):
    """Check fMRI data quality"""
    print(f"\n{'='*60}")
    print(f"Data Quality Report")
    print(f"{'='*60}")

    data = sio.loadmat(fmri_path)

    brain_data = data['dat']
    labels = data['labels'].flatten()
    runs = data['Run'].flatten()

    print(f"\nData shape: {brain_data.shape}")
    print(f"  Samples: {len(labels)}")
    print(f"  Voxels: {brain_data.shape[1]}")
    print(f"  Runs: {len(np.unique(runs))}")
    print(f"  Unique stimuli: {len(np.unique(labels))}")
    print(f"  Repetitions/stimulus: {len(labels) / len(np.unique(labels)):.1f}")

    # Check for NaN/Inf
    nan_count = np.isnan(brain_data).sum()
    inf_count = np.isinf(brain_data).sum()
    print(f"\nData integrity:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")

    # Check signal quality
    print(f"\nSignal statistics:")
    print(f"  Mean: {brain_data.mean():.6f}")
    print(f"  Std: {brain_data.std():.6f}")
    print(f"  Min: {brain_data.min():.6f}")
    print(f"  Max: {brain_data.max():.6f}")

    # Recommendations
    print(f"\n{'='*60}")
    print(f"Recommendations")
    print(f"{'='*60}")

    issues = []
    if brain_data.shape[0] < 500:
        issues.append("⚠️  Low sample count (<500). Consider collecting more data.")
    else:
        print("✅ Sample count looks good")

    if brain_data.shape[1] < 10000:
        issues.append("⚠️  Few voxels (<10000). Check preprocessing/masking.")
    else:
        print("✅ Voxel count looks good")

    if nan_count > 0 or inf_count > 0:
        issues.append("❌ Data contains NaN/Inf. Clean data before training!")
    else:
        print("✅ Data integrity OK")

    reps_per_stim = len(labels) / len(np.unique(labels))
    if reps_per_stim < 10:
        issues.append(f"⚠️  Few repetitions ({reps_per_stim:.1f} < 10). More is better.")
    else:
        print("✅ Repetition count looks good")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All checks passed! Data ready for training.")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare new data for MindCaptioning training'
    )
    parser.add_argument('--mode', choices=['fmri', 'features', 'check'], required=True,
                       help='Operation mode')
    parser.add_argument('--subject', type=str, help='Subject ID')
    parser.add_argument('--fmri-files', nargs='+', help='fMRI nifti files')
    parser.add_argument('--labels-file', type=str, help='Labels file (numpy .npy)')
    parser.add_argument('--runs-file', type=str, help='Runs file (numpy .npy)')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--captions-file', type=str, help='Captions CSV file')
    parser.add_argument('--model', type=str, default='microsoft/deberta-large',
                       help='Language model name')
    parser.add_argument('--check-file', type=str, help='fMRI file to check quality')

    args = parser.parse_args()

    if args.mode == 'fmri':
        # Preprocess fMRI data
        if not all([args.subject, args.fmri_files, args.labels_file,
                   args.runs_file, args.output]):
            print("Error: fmri mode requires --subject, --fmri-files, "
                  "--labels-file, --runs-file, --output")
            sys.exit(1)

        labels = np.load(args.labels_file)
        runs = np.load(args.runs_file)

        preprocess_fmri_data(
            subject_id=args.subject,
            fmri_files=args.fmri_files,
            labels=labels,
            runs=runs,
            output_path=args.output
        )

    elif args.mode == 'features':
        # Extract semantic features
        if not all([args.captions_file, args.output]):
            print("Error: features mode requires --captions-file, --output")
            sys.exit(1)

        captions_df = pd.read_csv(args.captions_file)
        captions = captions_df['caption'].tolist()

        extract_semantic_features(
            captions=captions,
            model_name=args.model,
            output_dir=args.output
        )

    elif args.mode == 'check':
        # Check data quality
        if not args.check_file:
            print("Error: check mode requires --check-file")
            sys.exit(1)

        check_data_quality(args.check_file)


if __name__ == "__main__":
    main()
