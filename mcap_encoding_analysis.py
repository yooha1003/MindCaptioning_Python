#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mind Captioning - Encoding Analysis (Python Port)

This script performs encoding analysis for the Mind Captioning project.
Converted from MATLAB to Python.

Original work:
    Horikawa, T. (2025) Mind captioning: Evolving descriptive text of mental
    content from human brain activity. Science Advances.

Python port by: Claude
Date: 2025-11-16

Usage:
    python mcap_encoding_analysis.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add util path
sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))
from mcap_config import McapConfig
from mcap_encoding import cv_encoding, gen_encoding
from thutil4 import setdir

class EncodingAnalysis:
    """Main class for encoding analysis"""

    def __init__(self, root_path='./'):
        """Initialize encoding analysis

        Parameters:
        -----------
        root_path : str
            Root path to MindCaptioning directory
        """
        self.root_path = root_path
        self.analysis_type = 'encoding'

        # Initialize configuration
        self.config = McapConfig(root_path, self.analysis_type)
        self.params = self.config.get_params()

        print("="*60)
        print("Mind Captioning - Encoding Analysis")
        print("="*60)
        print(f"Root path: {self.root_path}")
        print(f"Analysis type: {self.analysis_type}")
        print("="*60)

    def run_cv_encoding(self):
        """Perform cross-validation encoding analysis

        This analysis evaluates encoding models using cross-validation
        on the training data to assess how well brain activity can predict
        semantic features extracted by language models.
        """
        print("\n" + "="*60)
        print("Starting Cross-Validation Encoding Analysis")
        print("="*60)

        cv_encoding(self.params)

        print("\n" + "="*60)
        print("Cross-Validation Encoding Analysis Completed")
        print("="*60)

    def run_gen_encoding(self):
        """Perform generalization encoding analysis

        This analysis trains encoding models on training data and
        evaluates them on held-out test data to assess generalization
        performance.
        """
        print("\n" + "="*60)
        print("Starting Generalization Encoding Analysis")
        print("="*60)

        gen_encoding(self.params)

        print("\n" + "="*60)
        print("Generalization Encoding Analysis Completed")
        print("="*60)

    def run_all(self):
        """Run both CV and generalization encoding analyses"""
        self.run_cv_encoding()
        self.run_gen_encoding()
        print("\n" + "="*60)
        print("All Encoding Analyses Completed Successfully")
        print("="*60)


def main():
    """Main function"""
    # Initialize analysis
    analysis = EncodingAnalysis(root_path='./')

    # Perform cross-validation encoding analyses
    analysis.run_cv_encoding()

    # Perform generalization encoding analyses
    analysis.run_gen_encoding()

    print("\nEnd encoding analysis")


if __name__ == "__main__":
    main()
