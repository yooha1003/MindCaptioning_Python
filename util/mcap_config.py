#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mind Captioning - Configuration Module

This module handles configuration parameters for Mind Captioning analysis.
Converted from MATLAB mcap_setParams.m to Python.

Original work:
    Horikawa, T. (2025) Mind captioning: Evolving descriptive text of mental
    content from human brain activity. Science Advances.

Python port by: Claude
Date: 2025-11-16
"""

import os
import numpy as np
from pathlib import Path
from thutil4 import setdir

class McapConfig:
    """Configuration class for Mind Captioning analysis"""

    def __init__(self, root_path='./', analysis_type='encoding'):
        """Initialize configuration

        Parameters:
        -----------
        root_path : str
            Root path to MindCaptioning directory
        analysis_type : str
            Type of analysis ('encoding' or 'decoding')
        """
        self.root_path = root_path
        self.analysis_type = analysis_type

        # Initialize parameters dictionary
        self.params = {}

        # Set basic parameters
        self._set_basic_params()

        # Set path information
        self._set_paths()

        # Set data information
        self._set_data_info()

        # Set analysis parameters
        self._set_analysis_params()

    def _set_basic_params(self):
        """Set basic analysis settings"""
        self.params['root_path'] = self.root_path
        self.params['analysis_type'] = self.analysis_type

        # Misc settings
        self.params['del'] = False  # Set True to clean log files
        self.params['check_chkfile'] = False  # If False, skip file check (fast)
        self.params['check_mode_res'] = False  # If True, check by result file
        self.params['integrate_res'] = True  # If True, integrate results

    def _set_paths(self):
        """Set directory paths"""
        # Set directories
        self.params['savdir'] = setdir(
            os.path.join(self.root_path, 'res', self.analysis_type)
        )
        self.params['figdir'] = setdir(
            os.path.join(self.root_path, 'fig', self.analysis_type)
        )
        self.params['fmridir'] = setdir(
            os.path.join(self.root_path, 'data', 'fmri')
        )
        self.params['featdir'] = setdir(
            os.path.join(self.root_path, 'data', 'feature')
        )

    def _set_data_info(self):
        """Set data information"""
        # Subject IDs
        self.params['sbjID'] = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']

        # Model types
        self.params['model_types'] = ['deberta-large', 'timesformer']

    def _set_analysis_params(self):
        """Set analysis parameters"""
        # Cross-validation parameters
        cv_param = {}
        cv_param['run2fold_assign_idx'] = np.array([
            [1, 11, 21, 31, 41, 51],
            [10, 20, 30, 40, 50, 58]
        ])
        self.params['cparam'] = {'cv': cv_param}

        # Regularization parameters
        # [10, 1, 6] are the actual parameters used in the manuscript
        l2_param = {}
        l2_param['nparam_log_search'] = 10  # Can reduce for faster computation
        l2_param['lowL'] = 1
        l2_param['highL'] = 6

        # Demo mode (faster but with minor differences from manuscript results)
        do_demo = False  # Set True for demo mode
        if do_demo:
            # [4, 4, 5] can be used for demo (faster)
            l2_param['nparam_log_search'] = 4
            l2_param['lowL'] = 4
            l2_param['highL'] = 5

        self.params['aparam'] = {'l2': l2_param}

        # Decoding analysis parameters
        # n = 50000 in the study. Can reduce for faster computation (e.g., 5000)
        self.params['n_select_voxels'] = 50000

        # ROI settings
        roi_param = {}
        roi_param['roi_types'] = ['WB', 'WBnoLang', 'WBnoSem', 'WBnoVis', 'Lang']
        roi_param['cv_dec_skip_roi_types'] = ['WBnoLang', 'WBnoSem', 'WBnoVis', 'Lang']
        roi_param['gen_dec_skip_roi_types'] = []
        roi_param['language'] = ['temporal_language', 'frontal_language']
        self.params['rparam'] = roi_param

        # Feature parameters
        fparam = {}
        fparam['feature_path_template'] = os.path.join(
            self.root_path, 'data', 'feature', 'video', '{model_type}', 'layer*.mat'
        )
        self.params['fparam'] = fparam

        # Misc parameters
        misc_param = {}
        misc_param['dec_skip_models'] = ['timesformer']
        self.params['misc'] = misc_param

    def get_params(self):
        """Get all parameters

        Returns:
        --------
        dict
            Dictionary containing all parameters
        """
        return self.params

    def update_params(self, **kwargs):
        """Update parameters

        Parameters:
        -----------
        **kwargs : dict
            Parameters to update
        """
        self.params.update(kwargs)

    def print_params(self):
        """Print current parameters"""
        print("\n" + "="*60)
        print("Current Configuration Parameters")
        print("="*60)
        for key, value in self.params.items():
            print(f"{key}: {value}")
        print("="*60)
