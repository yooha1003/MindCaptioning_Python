#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mind Captioning - Decoding Analysis (Python Port)

This script performs decoding analysis for the Mind Captioning project.
Converted from MATLAB to Python.

Original work:
    Horikawa, T. (2025) Mind captioning: Evolving descriptive text of mental
    content from human brain activity. Science Advances.

Python port by: Claude
Date: 2025-11-16

Usage:
    python mcap_decoding_analysis.py [--cv]
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add util path
sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))
from mcap_config import McapConfig
from mcap_decoding import cv_decoding, gen_decoding
from thutil4 import setdir

class DecodingAnalysis:
    """Main class for decoding analysis"""

    def __init__(self, root_path='./'):
        """Initialize decoding analysis

        Parameters:
        -----------
        root_path : str
            Root path to MindCaptioning directory
        """
        self.root_path = root_path
        self.analysis_type = 'decoding'

        # Initialize configuration
        self.config = McapConfig(root_path, self.analysis_type)
        self.params = self.config.get_params()

        print("="*60)
        print("Mind Captioning - Decoding Analysis")
        print("="*60)
        print(f"Root path: {self.root_path}")
        print(f"Analysis type: {self.analysis_type}")
        print("="*60)

    def run_cv_decoding(self):
        """Perform cross-validation decoding analysis

        This analysis is time-consuming. Perform this only if you are
        interested in the results of validation and results in
        Extended Data Fig.7e.

        The cross-validation decoding evaluates how well we can decode
        semantic features from brain activity using cross-validation.
        """
        print("\n" + "="*60)
        print("Starting Cross-Validation Decoding Analysis")
        print("WARNING: This analysis is time-consuming!")
        print("="*60)

        cv_decoding(self.params)

        print("\n" + "="*60)
        print("Cross-Validation Decoding Analysis Completed")
        print("="*60)

    def run_gen_decoding(self):
        """Perform generalization decoding analysis

        This analysis trains decoding models on training data and
        decodes semantic features from test brain activity. These
        decoded features are then used for text generation.
        """
        print("\n" + "="*60)
        print("Starting Generalization Decoding Analysis")
        print("="*60)

        gen_decoding(self.params)

        print("\n" + "="*60)
        print("Generalization Decoding Analysis Completed")
        print("="*60)

    def run_all(self, do_cv=False):
        """Run decoding analyses

        Parameters:
        -----------
        do_cv : bool
            If True, run cross-validation decoding (time-consuming)
        """
        if do_cv:
            self.run_cv_decoding()

        self.run_gen_decoding()

        print("\n" + "="*60)
        print("All Decoding Analyses Completed Successfully")
        print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Mind Captioning - Decoding Analysis'
    )
    parser.add_argument(
        '--cv',
        action='store_true',
        help='Perform cross-validation decoding (time-consuming)'
    )
    args = parser.parse_args()

    # Initialize analysis
    analysis = DecodingAnalysis(root_path='./')

    # Perform cross-validation decoding analysis (optional)
    # cv decoding is time consuming.
    # Perform this only if you are interested in the results of validation
    # and results in Extended Data Fig.7e.
    do_cv = args.cv
    if do_cv:
        analysis.run_cv_decoding()

    # Perform generalization decoding analysis
    analysis.run_gen_decoding()

    print("\nEnd decoding analysis")


if __name__ == "__main__":
    main()
