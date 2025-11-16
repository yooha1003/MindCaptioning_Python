#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mind Captioning - Decoding Module

This module implements decoding analysis functions.
Converted from MATLAB to Python.

Original work:
    Horikawa, T. (2025) Mind captioning: Evolving descriptive text of mental
    content from human brain activity. Science Advances.

Python port by: Claude
Date: 2025-11-16
"""

import os
import sys
import numpy as np
import scipy.io as sio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from thutil4 import setdir
from mcap_utils import (
    load_data_wrapper,
    get_feature_params,
    generate_combinations,
    get_memory_available,
    save_chkfile,
    compute_correlation,
    select_voxels
)


def cv_decoding(params):
    """Perform cross-validation decoding analysis

    This function performs cross-validation decoding analysis to evaluate
    how well we can decode semantic features from brain activity.

    Parameters:
    -----------
    params : dict
        Dictionary containing all analysis parameters
    """
    print("\n" + "-"*60)
    print("Cross-Validation Decoding Analysis")
    print("WARNING: This is time-consuming!")
    print("-"*60)

    # Settings
    thmem = 50  # Memory threshold [GB]
    data_type = 'trainPerception'
    alg_type = 'l2'

    # Extract parameters
    root_path = params['root_path']
    sav_dir = params['savdir']
    fmri_dir = params['fmridir']
    encoding_dir = os.path.join(params['root_path'], 'res', 'encoding')
    sbj_ids = params['sbjID']
    model_types = params['model_types']
    roi_types = params['rparam']['roi_types']
    skip_roi_types = params['rparam']['cv_dec_skip_roi_types']
    n_select_voxels = params['n_select_voxels']

    # Skip certain model types
    model_types = [m for m in model_types if m not in params['misc']['dec_skip_models']]

    # Create condition combinations
    vars_list = [model_types, sbj_ids, roi_types]
    cond_names = ['model_type', 'sbj', 'roi_type']
    cond_list = generate_combinations(vars_list, randomize=True)

    # Process each condition
    for cond in cond_list:
        model_type, sbj, roi_type = cond

        # Skip certain ROI types
        if roi_type in skip_roi_types:
            continue

        print(f"\nProcessing: {model_type} - {sbj} - {roi_type}")

        # Get feature parameters
        feat_types, n_layers = get_feature_params(root_path, params['fparam'], model_type)
        if not feat_types:
            continue

        # Process each layer
        for feat_type in feat_types:
            suffix = f"{data_type}/{model_type}/{sbj}/{roi_type}/"
            save_fname_chk = os.path.join(sav_dir, suffix, f"{feat_type}_log.txt")
            save_fname = os.path.join(sav_dir, suffix, f"{feat_type}.mat")

            # Check if already processed
            check_file = save_fname if params['check_mode_res'] else save_fname_chk

            if os.path.exists(check_file):
                print(f"Skip: {check_file}")
                continue

            # Skip if memory is low
            if get_memory_available() < thmem * 1000000:
                print(f"Skip (low memory): {check_file}")
                continue

            print(f"Start: {save_fname_chk}")
            setdir(os.path.dirname(save_fname_chk))
            save_chkfile(save_fname_chk)

            # Perform CV decoding
            result = perform_cv_decoding_layer(
                sbj, model_type, feat_type, roi_type,
                data_type, fmri_dir, encoding_dir, root_path,
                params, n_select_voxels
            )

            if result is not None:
                print(f"Saving: {save_fname}")
                setdir(os.path.dirname(save_fname))
                sio.savemat(save_fname, result)

    print("\nCV decoding analysis completed")


def gen_decoding(params):
    """Perform generalization decoding analysis

    This function trains decoding models and decodes semantic features
    from test brain activity.

    Parameters:
    -----------
    params : dict
        Dictionary containing all analysis parameters
    """
    print("\n" + "-"*60)
    print("Generalization Decoding Analysis")
    print("-"*60)

    # Settings
    thmem = 50  # Memory threshold [GB]
    train_type = 'trainPerception'
    test_types = ['testPerception', 'testImagery']
    alg_type = 'l2'

    # Extract parameters
    root_path = params['root_path']
    sav_dir = params['savdir']
    fmri_dir = params['fmridir']
    encoding_dir = os.path.join(params['root_path'], 'res', 'encoding')
    sbj_ids = params['sbjID']
    model_types = params['model_types']
    roi_types = params['rparam']['roi_types']
    skip_roi_types = params['rparam']['gen_dec_skip_roi_types']
    n_select_voxels = params['n_select_voxels']

    # Skip certain model types
    model_types = [m for m in model_types if m not in params['misc']['dec_skip_models']]

    # Create condition combinations
    vars_list = [model_types, test_types, sbj_ids, roi_types]
    cond_names = ['model_type', 'test_type', 'sbj', 'roi_type']
    cond_list = generate_combinations(vars_list, randomize=True)

    # Process each condition
    for cond in cond_list:
        model_type, test_type, sbj, roi_type = cond

        # Skip certain ROI types
        if roi_type in skip_roi_types:
            continue

        print(f"\nProcessing: {model_type} - {test_type} - {sbj} - {roi_type}")

        # Get feature parameters
        feat_types, n_layers = get_feature_params(root_path, params['fparam'], model_type)
        if not feat_types:
            continue

        # Process each layer
        for feat_type in feat_types:
            suffix = f"{test_type}/{model_type}/{sbj}/{roi_type}/"
            save_fname_chk = os.path.join(sav_dir, suffix, f"{feat_type}_log.txt")
            save_fname = os.path.join(sav_dir, suffix, f"{feat_type}.mat")

            # Check if already processed
            check_file = save_fname if params['check_mode_res'] else save_fname_chk

            if os.path.exists(check_file):
                print(f"Skip: {check_file}")
                continue

            # Skip if memory is low
            if get_memory_available() < thmem * 1000000:
                print(f"Skip (low memory): {check_file}")
                continue

            print(f"Start: {save_fname_chk}")
            setdir(os.path.dirname(save_fname_chk))
            save_chkfile(save_fname_chk)

            # Perform generalization decoding
            result = perform_gen_decoding_layer(
                sbj, model_type, feat_type, roi_type,
                train_type, test_type, fmri_dir, encoding_dir,
                root_path, params, n_select_voxels
            )

            if result is not None:
                print(f"Saving: {save_fname}")
                setdir(os.path.dirname(save_fname))
                sio.savemat(save_fname, result)

    print("\nGeneralization decoding analysis completed")


def perform_cv_decoding_layer(sbj, model_type, feat_type, roi_type,
                               data_type, fmri_dir, encoding_dir,
                               root_path, params, n_select_voxels):
    """Perform CV decoding for a single layer

    Returns:
    --------
    dict or None
        Results dictionary or None if failed
    """
    try:
        # Load fMRI data
        dpath = os.path.join(fmri_dir, 'preprocessed', f"{data_type}_{sbj}.mat")
        brain_data, meta_info, labels, _, _, n_vox, n_sample = \
            load_data_wrapper(dpath, condition=1)

        # Load features
        fpath = os.path.join(root_path, 'data', 'feature', 'video', model_type, f"{feat_type}.mat")
        feat_data = sio.loadmat(fpath)
        features = feat_data['feat'][labels, :]

        # Load encoding results for voxel selection
        enc_path = os.path.join(
            encoding_dir, data_type, model_type, sbj, f"{feat_type}.mat"
        )
        enc_result = sio.loadmat(enc_path)

        # Select voxels based on encoding accuracy
        selected_voxels = select_voxels(
            enc_result, n_select_voxels, roi_type, root_path
        )

        # Subset brain data to selected voxels
        brain_data_selected = brain_data[:, selected_voxels]

        # Perform CV decoding
        cp = params['cparam']['cv']
        n_folds = cp['run2fold_assign_idx'].shape[1]
        predictions = np.zeros_like(features)
        correlations = np.zeros(n_folds)

        for fold_idx in range(n_folds):
            # Get train/test split
            run_start = cp['run2fold_assign_idx'][0, fold_idx]
            run_end = cp['run2fold_assign_idx'][1, fold_idx]

            test_idx = np.where(
                (meta_info['Run'] >= run_start) & (meta_info['Run'] <= run_end)
            )[0]
            train_idx = np.setdiff1d(np.arange(n_sample), test_idx)

            # Train model (brain -> features)
            # Use Ridge regression for stability
            model = Ridge(alpha=1.0)  # Could optimize this parameter
            model.fit(brain_data_selected[train_idx], features[train_idx])

            # Predict on test set
            predictions[test_idx] = model.predict(brain_data_selected[test_idx])

            # Compute correlation
            correlations[fold_idx] = compute_correlation(
                features[test_idx], predictions[test_idx]
            )

        return {
            'predictions': predictions,
            'correlations': correlations,
            'selected_voxels': selected_voxels,
            'mean_correlation': np.mean(correlations)
        }

    except Exception as e:
        print(f"Error in CV decoding: {e}")
        return None


def perform_gen_decoding_layer(sbj, model_type, feat_type, roi_type,
                                train_type, test_type, fmri_dir, encoding_dir,
                                root_path, params, n_select_voxels):
    """Perform generalization decoding for a single layer

    Returns:
    --------
    dict or None
        Results dictionary or None if failed
    """
    try:
        # Load training data
        train_path = os.path.join(fmri_dir, 'preprocessed', f"{train_type}_{sbj}.mat")
        brain_train, meta_train, labels_tr, _, _, n_vox, _ = \
            load_data_wrapper(train_path, condition=1)

        # Load test data
        test_path = os.path.join(fmri_dir, 'preprocessed', f"{test_type}_{sbj}.mat")
        brain_test, meta_test, labels_te, _, _, _, n_sample_te = \
            load_data_wrapper(test_path, condition=1)

        # Load features
        fpath = os.path.join(root_path, 'data', 'feature', 'video', model_type, f"{feat_type}.mat")
        feat_data = sio.loadmat(fpath)
        feat_train = feat_data['feat'][labels_tr, :]
        feat_test = feat_data['feat'][labels_te, :]

        # Load encoding results for voxel selection
        enc_path = os.path.join(
            encoding_dir, train_type, model_type, sbj, f"{feat_type}.mat"
        )
        enc_result = sio.loadmat(enc_path)

        # Select voxels based on encoding accuracy
        selected_voxels = select_voxels(
            enc_result, n_select_voxels, roi_type, root_path
        )

        # Subset brain data to selected voxels
        brain_train_selected = brain_train[:, selected_voxels]
        brain_test_selected = brain_test[:, selected_voxels]

        # Train decoding model on all training data
        model = Ridge(alpha=1.0)  # Could optimize this parameter
        model.fit(brain_train_selected, feat_train)

        # Decode features from test brain activity
        decoded_features = model.predict(brain_test_selected)

        # Compute correlation with actual features
        test_correlation = compute_correlation(feat_test, decoded_features)

        # Save decoded features for text generation
        return {
            'decoded_features': decoded_features,
            'test_correlation': test_correlation,
            'selected_voxels': selected_voxels,
            'weights': model.coef_,
            'video_idx': labels_te  # For tracking which videos these correspond to
        }

    except Exception as e:
        print(f"Error in generalization decoding: {e}")
        return None
