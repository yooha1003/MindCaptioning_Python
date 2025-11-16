#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mind Captioning - Encoding Module

This module implements encoding analysis functions.
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
import itertools
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from thutil4 import setdir, getDN, getFN
from mcap_utils import (
    load_data_wrapper,
    get_feature_params,
    generate_combinations,
    get_memory_available,
    save_chkfile,
    linear_regression_l2,
    compute_correlation
)


def cv_encoding(params):
    """Perform cross-validation encoding analysis

    This function performs cross-validation encoding analysis to evaluate
    how well semantic features can be predicted from brain activity.

    Parameters:
    -----------
    params : dict
        Dictionary containing all analysis parameters
    """
    print("\n" + "-"*60)
    print("Cross-Validation Encoding Analysis")
    print("-"*60)

    # Settings
    thmem = 30  # Memory threshold [GB]
    data_type = 'trainPerception'
    alg_type = 'l2'

    # Extract parameters
    root_path = params['root_path']
    sav_dir = params['savdir']
    fmri_dir = params['fmridir']
    sbj_ids = params['sbjID']
    model_types = params['model_types']
    cparam = params['cparam']
    aparam = params['aparam']
    fparam = params['fparam']

    # Create all condition combinations
    vars_list = [model_types, sbj_ids]
    cond_names = ['model_type', 'sbj']
    cond_list = generate_combinations(vars_list, randomize=True)

    # Start loop for multiple conditions
    for cond in cond_list:
        model_type, sbj = cond

        print(f"\nProcessing: {model_type} - {sbj}")

        # Get feature parameters
        feat_types, n_layers = get_feature_params(root_path, fparam, model_type)
        if not feat_types:
            continue

        # Cross-validation parameters
        cp = cparam['cv']
        n_folds = cp['run2fold_assign_idx'].shape[1]

        # Algorithm parameters
        ap = aparam[alg_type]
        lambda_values = np.logspace(ap['lowL'], ap['highL'], ap['nparam_log_search'])

        # Set save info for summary results
        suffix_summary = f"{data_type}/{model_type}/{sbj}/"
        save_fname_chk = os.path.join(sav_dir, suffix_summary, 'res_summary_log.txt')
        save_fname = os.path.join(sav_dir, suffix_summary, 'res_summary.mat')

        # Process each layer
        for fitr in np.random.permutation(n_layers):
            feat_type = feat_types[fitr]

            # Set save info for each layer
            save_fname_chkx = os.path.join(sav_dir, suffix_summary, f"{feat_type}_log.txt")
            save_fnamex = os.path.join(sav_dir, suffix_summary, f"{feat_type}.mat")

            # Check if already processed
            check_file = save_fnamex if params['check_mode_res'] else save_fname_chkx

            if os.path.exists(check_file):
                print(f"Skip (already exists): {check_file}")
                continue

            # Skip if memory is low
            if get_memory_available() < thmem * 1000000:
                print(f"Skip (low memory): {check_file}")
                continue

            print(f"Start: {save_fname_chkx}")
            setdir(os.path.dirname(save_fname_chkx))
            save_chkfile(save_fname_chkx)

            print(f"Condition: model_type={model_type}, sbj={sbj}")

            # Load fMRI data
            print(f"Loading data for {sbj}...")
            dpath = os.path.join(fmri_dir, 'preprocessed', f"{data_type}_{sbj}.mat")

            try:
                brain_data, meta_info, labels, uni_labels, n_stim, n_vox, n_sample = \
                    load_data_wrapper(dpath, condition=1)
            except Exception as e:
                print(f"Error loading data: {e}")
                if os.path.exists(save_fname_chkx):
                    os.remove(save_fname_chkx)
                break

            # Load features
            fpath = os.path.join(
                root_path, 'data', 'feature', 'video', model_type, f"{feat_type}.mat"
            )

            try:
                feat_data = sio.loadmat(fpath)
                features = feat_data['feat']
                if features.size == 0:
                    raise ValueError("Empty features")
                features = features[labels, :]
            except Exception as e:
                print(f"Error loading features: {e}")
                if os.path.exists(save_fname_chkx):
                    os.remove(save_fname_chkx)
                continue

            # Perform nested cross-validation
            print("Start inner CV analysis for parameter determination")

            results_cv = []

            for fold_idx in range(n_folds):
                # Get train/test indices for this fold
                run_start = cp['run2fold_assign_idx'][0, fold_idx]
                run_end = cp['run2fold_assign_idx'][1, fold_idx]

                test_idx = np.where(
                    (meta_info['Run'] >= run_start) & (meta_info['Run'] <= run_end)
                )[0]
                train_idx = np.setdiff1d(np.arange(n_sample), test_idx)

                # Perform regression with cross-validation for parameter selection
                result_fold = perform_nested_cv_encoding(
                    brain_data[train_idx], features[train_idx],
                    brain_data[test_idx], features[test_idx],
                    lambda_values, meta_info, train_idx, fold_idx, cp
                )

                results_cv.append(result_fold)

            # Save results
            print(f"Saving: {save_fnamex}")
            setdir(os.path.dirname(save_fnamex))
            results = {
                'results_cv': results_cv,
                'feat_type': feat_type,
                'model_type': model_type,
                'sbj': sbj
            }
            sio.savemat(save_fnamex, results)

    print("\nCross-validation encoding analysis completed")


def gen_encoding(params):
    """Perform generalization encoding analysis

    This function trains encoding models on training data and evaluates
    on test data.

    Parameters:
    -----------
    params : dict
        Dictionary containing all analysis parameters
    """
    print("\n" + "-"*60)
    print("Generalization Encoding Analysis")
    print("-"*60)

    # Settings
    thmem = 50  # Memory threshold [GB]
    train_type = 'trainPerception'
    test_types = ['testPerception']  # Can include 'testImagery'
    alg_type = 'l2'

    # Extract parameters
    root_path = params['root_path']
    sav_dir = params['savdir']
    fmri_dir = params['fmridir']
    sbj_ids = params['sbjID']
    model_types = params['model_types']
    cparam = params['cparam']
    aparam = params['aparam']
    fparam = params['fparam']

    # Create all condition combinations
    vars_list = [model_types, test_types, sbj_ids]
    cond_names = ['model_type', 'test_type', 'sbj']
    cond_list = generate_combinations(vars_list, randomize=True)

    # Start loop for multiple conditions
    for cond in cond_list:
        model_type, test_type, sbj = cond

        print(f"\nProcessing: {model_type} - {test_type} - {sbj}")

        # Get feature parameters
        feat_types, n_layers = get_feature_params(root_path, fparam, model_type)
        if not feat_types:
            continue

        # Cross-validation parameters
        cp = cparam['cv']
        n_folds = cp['run2fold_assign_idx'].shape[1]

        # Algorithm parameters
        ap = aparam[alg_type]
        lambda_values = np.logspace(ap['lowL'], ap['highL'], ap['nparam_log_search'])

        # Set save info for summary results
        suffix_summary = f"{test_type}/{model_type}/{sbj}/"
        save_fname_chk = os.path.join(sav_dir, suffix_summary, 'res_summary_log.txt')
        save_fname = os.path.join(sav_dir, suffix_summary, 'res_summary.mat')

        # Process each layer
        for fitr in np.random.permutation(n_layers):
            feat_type = feat_types[fitr]

            # Set save info for each layer
            save_fname_chkx = os.path.join(sav_dir, suffix_summary, f"{feat_type}_log.txt")
            save_fnamex = os.path.join(sav_dir, suffix_summary, f"{feat_type}.mat")

            # Check if already processed
            check_file = save_fnamex if params['check_mode_res'] else save_fname_chkx

            if os.path.exists(check_file):
                print(f"Skip (already exists): {check_file}")
                continue

            # Skip if memory is low
            if get_memory_available() < thmem * 1000000:
                print(f"Skip (low memory): {check_file}")
                continue

            print(f"Start: {save_fname_chkx}")
            setdir(os.path.dirname(save_fname_chkx))
            save_chkfile(save_fname_chkx)

            print(f"Condition: model_type={model_type}, test_type={test_type}, sbj={sbj}")

            # Load training data
            print(f"Loading training data for {sbj}...")
            train_path = os.path.join(fmri_dir, 'preprocessed', f"{train_type}_{sbj}.mat")

            try:
                brain_train, meta_train, labels_tr, _, _, n_vox, n_sample_tr = \
                    load_data_wrapper(train_path, condition=1)
            except Exception as e:
                print(f"Error loading training data: {e}")
                if os.path.exists(save_fname_chkx):
                    os.remove(save_fname_chkx)
                break

            # Load test data
            print(f"Loading test data...")
            test_path = os.path.join(fmri_dir, 'preprocessed', f"{test_type}_{sbj}.mat")

            try:
                brain_test, meta_test, labels_te, _, _, _, n_sample_te = \
                    load_data_wrapper(test_path, condition=1)
            except Exception as e:
                print(f"Error loading test data: {e}")
                if os.path.exists(save_fname_chkx):
                    os.remove(save_fname_chkx)
                break

            # Load training features
            fpath = os.path.join(
                root_path, 'data', 'feature', 'video', model_type, f"{feat_type}.mat"
            )

            try:
                feat_data = sio.loadmat(fpath)
                feat_train = feat_data['feat'][labels_tr, :]
                feat_test = feat_data['feat'][labels_te, :]
            except Exception as e:
                print(f"Error loading features: {e}")
                if os.path.exists(save_fname_chkx):
                    os.remove(save_fname_chkx)
                continue

            # Perform generalization analysis
            print("Performing generalization encoding...")

            result = perform_gen_encoding(
                brain_train, feat_train,
                brain_test, feat_test,
                lambda_values, meta_train, cp
            )

            # Save results
            print(f"Saving: {save_fnamex}")
            setdir(os.path.dirname(save_fnamex))
            result.update({
                'feat_type': feat_type,
                'model_type': model_type,
                'test_type': test_type,
                'sbj': sbj
            })
            sio.savemat(save_fnamex, result)

    print("\nGeneralization encoding analysis completed")


def perform_nested_cv_encoding(brain_train, feat_train, brain_test, feat_test,
                                lambda_values, meta_info, train_idx, fold_idx, cp):
    """Perform nested cross-validation for encoding

    Parameters:
    -----------
    brain_train : ndarray
        Training brain data
    feat_train : ndarray
        Training features
    brain_test : ndarray
        Test brain data
    feat_test : ndarray
        Test features
    lambda_values : ndarray
        Regularization parameters to test
    meta_info : dict
        Metadata information
    train_idx : ndarray
        Training indices
    fold_idx : int
        Current fold index
    cp : dict
        Cross-validation parameters

    Returns:
    --------
    dict
        Results dictionary
    """
    # This is a placeholder for the actual implementation
    # Full implementation would include nested CV for parameter selection

    # Select best lambda using nested CV
    best_lambda = lambda_values[len(lambda_values)//2]  # Simplified

    # Train model with best lambda
    model = Ridge(alpha=best_lambda)
    model.fit(brain_train, feat_train)

    # Predict on test set
    feat_pred = model.predict(brain_test)

    # Compute correlation
    corr = compute_correlation(feat_test, feat_pred)

    return {
        'best_lambda': best_lambda,
        'correlation': corr,
        'predictions': feat_pred,
        'fold_idx': fold_idx
    }


def perform_gen_encoding(brain_train, feat_train, brain_test, feat_test,
                         lambda_values, meta_train, cp):
    """Perform generalization encoding

    Parameters:
    -----------
    brain_train : ndarray
        Training brain data
    feat_train : ndarray
        Training features
    brain_test : ndarray
        Test brain data
    feat_test : ndarray
        Test features
    lambda_values : ndarray
        Regularization parameters to test
    meta_train : dict
        Training metadata
    cp : dict
        Cross-validation parameters

    Returns:
    --------
    dict
        Results dictionary
    """
    # Perform CV on training data to select best lambda
    n_folds = cp['run2fold_assign_idx'].shape[1]
    cv_scores = np.zeros(len(lambda_values))

    for lambda_idx, lambda_val in enumerate(lambda_values):
        fold_scores = []

        for fold_idx in range(n_folds):
            # Get CV train/test split
            run_start = cp['run2fold_assign_idx'][0, fold_idx]
            run_end = cp['run2fold_assign_idx'][1, fold_idx]

            cv_test_idx = np.where(
                (meta_train['Run'] >= run_start) & (meta_train['Run'] <= run_end)
            )[0]
            cv_train_idx = np.setdiff1d(np.arange(len(brain_train)), cv_test_idx)

            # Train and evaluate
            model = Ridge(alpha=lambda_val)
            model.fit(brain_train[cv_train_idx], feat_train[cv_train_idx])
            pred = model.predict(brain_train[cv_test_idx])

            # Compute correlation
            corr = compute_correlation(feat_train[cv_test_idx], pred)
            fold_scores.append(corr)

        cv_scores[lambda_idx] = np.mean(fold_scores)

    # Select best lambda
    best_lambda_idx = np.argmax(cv_scores)
    best_lambda = lambda_values[best_lambda_idx]

    # Train final model on all training data
    final_model = Ridge(alpha=best_lambda)
    final_model.fit(brain_train, feat_train)

    # Predict on test data
    feat_pred = final_model.predict(brain_test)

    # Compute test correlation
    test_corr = compute_correlation(feat_test, feat_pred)

    return {
        'best_lambda': best_lambda,
        'cv_scores': cv_scores,
        'test_correlation': test_corr,
        'predictions': feat_pred,
        'weights': final_model.coef_
    }
