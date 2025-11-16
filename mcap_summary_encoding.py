#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mind Captioning - Encoding Summary (Python Port)

This script generates summary figures and statistics for encoding analysis.
Converted from MATLAB to Python.

Original work:
    Horikawa, T. (2025) Mind captioning: Evolving descriptive text of mental
    content from human brain activity. Science Advances.

Python port by: Claude
Date: 2025-11-16

Usage:
    python mcap_summary_encoding.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Add util path
sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))
from mcap_config import McapConfig
from thutil4 import setdir


class EncodingSummary:
    """Class for summarizing encoding analysis results"""

    def __init__(self, root_path='./'):
        """Initialize encoding summary

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

        # Set up directories
        self.res_dir = os.path.join(root_path, 'res', 'encoding')
        self.fig_dir = os.path.join(root_path, 'fig', 'encoding')
        setdir(self.fig_dir)

        print("="*60)
        print("Mind Captioning - Encoding Summary")
        print("="*60)

    def load_results(self, data_type='trainPerception'):
        """Load encoding results

        Parameters:
        -----------
        data_type : str
            Data type to load

        Returns:
        --------
        dict
            Dictionary of results by subject and model
        """
        results = {}

        for sbj in self.params['sbjID']:
            results[sbj] = {}

            for model_type in self.params['model_types']:
                res_dir = os.path.join(self.res_dir, data_type, model_type, sbj)

                if not os.path.exists(res_dir):
                    continue

                # Load all layer results
                layer_files = sorted(glob.glob(os.path.join(res_dir, 'layer*.mat')))

                model_results = []
                for layer_file in layer_files:
                    try:
                        layer_data = sio.loadmat(layer_file)
                        model_results.append(layer_data)
                    except Exception as e:
                        print(f"Error loading {layer_file}: {e}")

                if model_results:
                    results[sbj][model_type] = model_results

        return results

    def summarize_accuracy(self, results):
        """Summarize encoding accuracy across subjects and models

        Parameters:
        -----------
        results : dict
            Results dictionary

        Returns:
        --------
        dict
            Summary statistics
        """
        summary = {
            'subjects': [],
            'models': [],
            'layers': [],
            'mean_corr': [],
            'std_corr': []
        }

        for sbj in results:
            for model_type in results[sbj]:
                for layer_idx, layer_result in enumerate(results[sbj][model_type]):
                    if 'correlations' in layer_result:
                        corr = layer_result['correlations']
                        if isinstance(corr, np.ndarray):
                            mean_corr = np.mean(corr)
                            std_corr = np.std(corr)

                            summary['subjects'].append(sbj)
                            summary['models'].append(model_type)
                            summary['layers'].append(layer_idx + 1)
                            summary['mean_corr'].append(mean_corr)
                            summary['std_corr'].append(std_corr)

        return summary

    def plot_accuracy(self, summary, save_path=None):
        """Plot encoding accuracy

        Parameters:
        -----------
        summary : dict
            Summary statistics
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Accuracy by layer
        ax = axes[0]
        for model in np.unique(summary['models']):
            model_idx = np.array(summary['models']) == model
            layers = np.array(summary['layers'])[model_idx]
            mean_corr = np.array(summary['mean_corr'])[model_idx]

            ax.plot(layers, mean_corr, marker='o', label=model)

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Encoding Accuracy (Correlation)', fontsize=12)
        ax.set_title('Encoding Accuracy by Layer', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Accuracy by subject
        ax = axes[1]
        subjects = np.unique(summary['subjects'])
        models = np.unique(summary['models'])

        x = np.arange(len(subjects))
        width = 0.35

        for i, model in enumerate(models):
            subj_means = []
            for sbj in subjects:
                idx = (np.array(summary['subjects']) == sbj) & (np.array(summary['models']) == model)
                if np.any(idx):
                    subj_means.append(np.mean(np.array(summary['mean_corr'])[idx]))
                else:
                    subj_means.append(0)

            ax.bar(x + i * width, subj_means, width, label=model)

        ax.set_xlabel('Subject', fontsize=12)
        ax.set_ylabel('Mean Encoding Accuracy', fontsize=12)
        ax.set_title('Encoding Accuracy by Subject', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(subjects)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {save_path}")

        plt.show()

    def plot_best_layer(self, summary, save_path=None):
        """Plot best encoding layer for each subject

        Parameters:
        -----------
        summary : dict
            Summary statistics
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        subjects = np.unique(summary['subjects'])
        models = np.unique(summary['models'])

        for model in models:
            best_layers = []
            for sbj in subjects:
                idx = (np.array(summary['subjects']) == sbj) & (np.array(summary['models']) == model)
                if np.any(idx):
                    corrs = np.array(summary['mean_corr'])[idx]
                    layers = np.array(summary['layers'])[idx]
                    best_layer = layers[np.argmax(corrs)]
                    best_layers.append(best_layer)
                else:
                    best_layers.append(0)

            ax.plot(subjects, best_layers, marker='o', markersize=10, label=model, linewidth=2)

        ax.set_xlabel('Subject', fontsize=12)
        ax.set_ylabel('Best Layer', fontsize=12)
        ax.set_title('Best Encoding Layer by Subject', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {save_path}")

        plt.show()

    def generate_figures(self, data_type='trainPerception'):
        """Generate all summary figures

        Parameters:
        -----------
        data_type : str
            Data type to summarize
        """
        print("\nLoading encoding results...")
        results = self.load_results(data_type)

        print("Summarizing accuracy...")
        summary = self.summarize_accuracy(results)

        if not summary['subjects']:
            print("No results found!")
            return

        print("Generating figures...")

        # Figure 1: Accuracy plots
        save_path1 = os.path.join(self.fig_dir, f'{data_type}_accuracy.png')
        self.plot_accuracy(summary, save_path1)

        # Figure 2: Best layer plot
        save_path2 = os.path.join(self.fig_dir, f'{data_type}_best_layer.png')
        self.plot_best_layer(summary, save_path2)

        print("\nAll figures generated successfully!")

    def print_summary_stats(self, data_type='trainPerception'):
        """Print summary statistics

        Parameters:
        -----------
        data_type : str
            Data type to summarize
        """
        results = self.load_results(data_type)
        summary = self.summarize_accuracy(results)

        print("\n" + "="*60)
        print("Encoding Summary Statistics")
        print("="*60)

        for sbj in np.unique(summary['subjects']):
            print(f"\n{sbj}:")
            for model in np.unique(summary['models']):
                idx = (np.array(summary['subjects']) == sbj) & (np.array(summary['models']) == model)
                if np.any(idx):
                    mean_corr = np.array(summary['mean_corr'])[idx]
                    print(f"  {model}:")
                    print(f"    Mean: {np.mean(mean_corr):.4f}")
                    print(f"    Max:  {np.max(mean_corr):.4f}")
                    print(f"    Best layer: {np.array(summary['layers'])[idx][np.argmax(mean_corr)]}")

        print("="*60)


def main():
    """Main function"""
    # Initialize summary
    summary = EncodingSummary(root_path='./')

    # Print summary statistics
    summary.print_summary_stats('trainPerception')

    # Generate figures
    summary.generate_figures('trainPerception')

    # Also process test data if available
    summary.print_summary_stats('testPerception')
    summary.generate_figures('testPerception')

    print("\nEnd encoding summary")


if __name__ == "__main__":
    main()
