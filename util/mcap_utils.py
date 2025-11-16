#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mind Captioning - Utility Functions

This module contains utility functions for Mind Captioning analysis.
Converted from MATLAB utility functions to Python.

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
import glob
import itertools
import psutil
from scipy import stats
from sklearn.linear_model import Ridge


def load_data_wrapper(data_path, condition=None):
    """Load fMRI data from .mat file

    Parameters:
    -----------
    data_path : str
        Path to data file
    condition : int, optional
        Condition to filter

    Returns:
    --------
    tuple
        (brain_data, meta_info, labels, uni_labels, n_stim, n_vox, n_sample)
    """
    data = sio.loadmat(data_path)

    brain_data = data['dat']  # Samples x Voxels
    meta_info = {}

    if 'Run' in data:
        meta_info['Run'] = data['Run'].flatten()
    if 'Condition' in data:
        meta_info['Condition'] = data['Condition'].flatten()
    if 'labels' in data:
        labels = data['labels'].flatten() - 1  # Convert to 0-indexed
    else:
        labels = np.arange(brain_data.shape[0])

    # Filter by condition if specified
    if condition is not None and 'Condition' in meta_info:
        cond_idx = meta_info['Condition'] == condition
        brain_data = brain_data[cond_idx]
        labels = labels[cond_idx]
        for key in meta_info:
            meta_info[key] = meta_info[key][cond_idx]

    uni_labels = np.unique(labels)
    n_stim = len(uni_labels)
    n_vox = brain_data.shape[1]
    n_sample = brain_data.shape[0]

    return brain_data, meta_info, labels, uni_labels, n_stim, n_vox, n_sample


def get_feature_params(root_path, fparam, model_type):
    """Get feature parameters for a model type

    Parameters:
    -----------
    root_path : str
        Root path
    fparam : dict
        Feature parameters
    model_type : str
        Model type

    Returns:
    --------
    tuple
        (feat_types, n_layers)
    """
    # Find all feature files for this model
    feature_dir = os.path.join(root_path, 'data', 'feature', 'video', model_type)

    if not os.path.exists(feature_dir):
        return [], 0

    # Find all layer files
    layer_files = sorted(glob.glob(os.path.join(feature_dir, 'layer*.mat')))

    if not layer_files:
        return [], 0

    # Extract layer names
    feat_types = [
        os.path.splitext(os.path.basename(f))[0]
        for f in layer_files
    ]

    n_layers = len(feat_types)

    return feat_types, n_layers


def generate_combinations(vars_list, randomize=False):
    """Generate all combinations of variables

    Parameters:
    -----------
    vars_list : list
        List of variable lists
    randomize : bool
        If True, randomize order

    Returns:
    --------
    list
        List of combinations
    """
    # Generate all combinations
    combinations = list(itertools.product(*vars_list))

    # Randomize if requested
    if randomize:
        np.random.shuffle(combinations)

    return combinations


def get_memory_available():
    """Get available memory in KB

    Returns:
    --------
    float
        Available memory in KB
    """
    mem = psutil.virtual_memory()
    # Return available memory in KB
    return mem.available / 1024


def save_chkfile(filepath):
    """Save a checkpoint file

    Parameters:
    -----------
    filepath : str
        Path to checkpoint file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write timestamp to file
    with open(filepath, 'w') as f:
        import datetime
        f.write(f"Started: {datetime.datetime.now()}\n")


def compute_correlation(y_true, y_pred, axis=None):
    """Compute correlation between true and predicted values

    Parameters:
    -----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
    axis : int, optional
        Axis along which to compute correlation

    Returns:
    --------
    float or ndarray
        Correlation value(s)
    """
    if axis is None:
        # Flatten and compute overall correlation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Remove NaN values
        valid_idx = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        if np.sum(valid_idx) == 0:
            return 0.0

        corr, _ = stats.pearsonr(y_true_flat[valid_idx], y_pred_flat[valid_idx])
        return corr
    else:
        # Compute correlation along specified axis
        if axis == 0:
            n_feat = y_true.shape[1]
            corrs = np.zeros(n_feat)
            for i in range(n_feat):
                valid_idx = ~(np.isnan(y_true[:, i]) | np.isnan(y_pred[:, i]))
                if np.sum(valid_idx) > 1:
                    corrs[i], _ = stats.pearsonr(
                        y_true[valid_idx, i], y_pred[valid_idx, i]
                    )
            return corrs
        else:
            raise ValueError("Only axis=0 is currently supported")


def linear_regression_l2(X_train, y_train, X_test, alpha=1.0):
    """Perform L2-regularized linear regression

    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training targets
    X_test : ndarray
        Test features
    alpha : float
        Regularization parameter

    Returns:
    --------
    ndarray
        Predictions on test set
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred


def select_voxels(encoding_result, n_select, roi_type, root_path):
    """Select top voxels based on encoding accuracy

    Parameters:
    -----------
    encoding_result : dict
        Encoding analysis results
    n_select : int
        Number of voxels to select
    roi_type : str
        ROI type
    root_path : str
        Root path

    Returns:
    --------
    ndarray
        Indices of selected voxels
    """
    # This is a simplified version
    # Full implementation would load ROI masks and select voxels accordingly

    if 'correlations' in encoding_result:
        # Get correlation for each voxel
        if encoding_result['correlations'].ndim > 1:
            voxel_corrs = np.mean(encoding_result['correlations'], axis=0)
        else:
            voxel_corrs = encoding_result['correlations']

        # Select top N voxels
        selected_idx = np.argsort(voxel_corrs)[-n_select:]

        return selected_idx
    else:
        # If no encoding results, select random voxels
        # In real implementation, would use all voxels in ROI
        print("Warning: No encoding correlations found, selecting random voxels")
        return np.arange(n_select)


def normalize_data(data, mu=None, sd=None):
    """Normalize data (z-score)

    Parameters:
    -----------
    data : ndarray
        Data to normalize
    mu : ndarray, optional
        Mean values (if None, compute from data)
    sd : ndarray, optional
        Standard deviation values (if None, compute from data)

    Returns:
    --------
    tuple
        (normalized_data, mu, sd)
    """
    if mu is None:
        mu = np.mean(data, axis=0, keepdims=True)
    if sd is None:
        sd = np.std(data, axis=0, keepdims=True)

    # Avoid division by zero
    sd = np.where(sd == 0, 1, sd)

    normalized_data = (data - mu) / sd

    return normalized_data, mu, sd


def compute_identification_accuracy(features_test, features_decoded, features_ref):
    """Compute identification accuracy

    Parameters:
    -----------
    features_test : ndarray
        Test features
    features_decoded : ndarray
        Decoded features
    features_ref : ndarray
        Reference features for all stimuli

    Returns:
    --------
    float
        Identification accuracy
    """
    n_test = features_decoded.shape[0]
    correct = 0

    for i in range(n_test):
        # Compute similarity between decoded and all reference features
        similarities = np.zeros(len(features_ref))
        for j in range(len(features_ref)):
            similarities[j] = compute_correlation(
                features_decoded[i:i+1], features_ref[j:j+1]
            )

        # Check if maximum similarity is for correct stimulus
        if np.argmax(similarities) == i:
            correct += 1

    accuracy = correct / n_test

    return accuracy


def evaluate_accuracy(y_true, y_pred, metric='correlation'):
    """Evaluate prediction accuracy

    Parameters:
    -----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
    metric : str
        Metric type ('correlation', 'mse', 'r2')

    Returns:
    --------
    float
        Accuracy score
    """
    if metric == 'correlation':
        return compute_correlation(y_true, y_pred)
    elif metric == 'mse':
        return np.mean((y_true - y_pred) ** 2)
    elif metric == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def load_roi_mask(roi_type, sbj, root_path):
    """Load ROI mask for a subject

    Parameters:
    -----------
    roi_type : str
        ROI type
    sbj : str
        Subject ID
    root_path : str
        Root path

    Returns:
    --------
    ndarray or None
        ROI mask indices
    """
    # This is a placeholder
    # Full implementation would load actual ROI masks
    roi_path = os.path.join(root_path, 'data', 'roi', sbj, f"{roi_type}.mat")

    if os.path.exists(roi_path):
        roi_data = sio.loadmat(roi_path)
        if 'voxel_idx' in roi_data:
            return roi_data['voxel_idx'].flatten() - 1  # Convert to 0-indexed
        elif 'mask' in roi_data:
            return np.where(roi_data['mask'].flatten())[0]

    return None


def merge_results(result_list):
    """Merge multiple result dictionaries

    Parameters:
    -----------
    result_list : list
        List of result dictionaries

    Returns:
    --------
    dict
        Merged results
    """
    if not result_list:
        return {}

    merged = {}

    # Get all keys from first result
    for key in result_list[0].keys():
        if isinstance(result_list[0][key], np.ndarray):
            # Stack arrays
            merged[key] = np.vstack([r[key] for r in result_list])
        elif isinstance(result_list[0][key], (int, float)):
            # Average scalars
            merged[key] = np.mean([r[key] for r in result_list])
        elif isinstance(result_list[0][key], list):
            # Concatenate lists
            merged[key] = [item for r in result_list for item in r[key]]
        else:
            # Keep first value for other types
            merged[key] = result_list[0][key]

    return merged
