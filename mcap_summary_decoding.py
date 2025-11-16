#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mind Captioning - Decoding Summary (Python Port)

This script generates summary figures and statistics for decoding and text generation analysis.
Converted from MATLAB to Python.

Original work:
    Horikawa, T. (2025) Mind captioning: Evolving descriptive text of mental
    content from human brain activity. Science Advances.

Python port by: Claude
Date: 2025-11-16

Usage:
    python mcap_summary_decoding.py
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


class DecodingSummary:
    """Class for summarizing decoding and text generation results"""

    def __init__(self, root_path='./'):
        """Initialize decoding summary

        Parameters:
        -----------
        root_path : str
            Root path to MindCaptioning directory
        """
        self.root_path = root_path

        # Set up directories
        self.dec_res_dir = os.path.join(root_path, 'res', 'decoding')
        self.txt_res_dir = os.path.join(root_path, 'res', 'text_generation')
        self.fig_dir = os.path.join(root_path, 'fig', 'decoding')
        setdir(self.fig_dir)

        print("="*60)
        print("Mind Captioning - Decoding & Text Generation Summary")
        print("="*60)

    def load_decoding_results(self, data_type='testPerception'):
        """Load decoding results

        Parameters:
        -----------
        data_type : str
            Data type to load

        Returns:
        --------
        dict
            Dictionary of results
        """
        results = {}

        sbj_ids = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        model_types = ['deberta-large']
        roi_types = ['WB', 'Lang']

        for sbj in sbj_ids:
            results[sbj] = {}

            for model_type in model_types:
                results[sbj][model_type] = {}

                for roi_type in roi_types:
                    res_dir = os.path.join(
                        self.dec_res_dir, data_type, model_type, sbj, roi_type
                    )

                    if not os.path.exists(res_dir):
                        continue

                    # Load results
                    layer_files = sorted(glob.glob(os.path.join(res_dir, 'layer*.mat')))

                    roi_results = []
                    for layer_file in layer_files:
                        try:
                            layer_data = sio.loadmat(layer_file)
                            roi_results.append(layer_data)
                        except Exception as e:
                            print(f"Error loading {layer_file}: {e}")

                    if roi_results:
                        results[sbj][model_type][roi_type] = roi_results

        return results

    def load_text_generation_results(self, data_type='testPerception'):
        """Load text generation results

        Parameters:
        -----------
        data_type : str
            Data type to load

        Returns:
        --------
        dict
            Dictionary of results
        """
        results = {}

        sbj_ids = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        mlm_type = 'roberta-large'
        lm_type = 'deberta-large'
        roi_types = ['WB', 'Lang']

        for sbj in sbj_ids:
            results[sbj] = {}

            for roi_type in roi_types:
                res_dir = os.path.join(
                    self.txt_res_dir, data_type,
                    f'mlm_{mlm_type}', f'lm_{lm_type}',
                    sbj, roi_type, 'res'
                )

                if not os.path.exists(res_dir):
                    continue

                # Load all result files
                res_files = sorted(glob.glob(os.path.join(res_dir, 'res_samp*.mat')))

                roi_results = []
                for res_file in res_files:
                    try:
                        data = sio.loadmat(res_file)
                        roi_results.append(data)
                    except Exception as e:
                        print(f"Error loading {res_file}: {e}")

                if roi_results:
                    results[sbj][roi_type] = roi_results

        return results

    def summarize_decoding_accuracy(self, results):
        """Summarize decoding accuracy

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
            'rois': [],
            'mean_corr': []
        }

        for sbj in results:
            for model in results[sbj]:
                for roi in results[sbj][model]:
                    for layer_result in results[sbj][model][roi]:
                        if 'test_correlation' in layer_result:
                            corr = float(layer_result['test_correlation'])
                            summary['subjects'].append(sbj)
                            summary['models'].append(model)
                            summary['rois'].append(roi)
                            summary['mean_corr'].append(corr)

        return summary

    def summarize_text_generation(self, results):
        """Summarize text generation results

        Parameters:
        -----------
        results : dict
            Results dictionary

        Returns:
        --------
        dict
            Summary with generated texts
        """
        summary = {
            'subjects': [],
            'rois': [],
            'sample_idx': [],
            'generated_text': [],
            'scores': []
        }

        for sbj in results:
            for roi in results[sbj]:
                for res in results[sbj][roi]:
                    if 'best_cands' in res:
                        best_text = res['best_cands'][-1] if isinstance(res['best_cands'], list) else ''
                        score = float(res['scores_all'][-1]) if 'scores_all' in res else 0.0
                        sample_idx = int(res['decsampidx']) if 'decsampidx' in res else 0

                        summary['subjects'].append(sbj)
                        summary['rois'].append(roi)
                        summary['sample_idx'].append(sample_idx)
                        summary['generated_text'].append(best_text)
                        summary['scores'].append(score)

        return summary

    def plot_decoding_accuracy(self, summary, save_path=None):
        """Plot decoding accuracy

        Parameters:
        -----------
        summary : dict
            Summary statistics
        save_path : str, optional
            Path to save figure
        """
        if not summary['subjects']:
            print("No decoding results to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        subjects = sorted(set(summary['subjects']))
        rois = sorted(set(summary['rois']))

        x = np.arange(len(subjects))
        width = 0.35

        for i, roi in enumerate(rois):
            subj_means = []
            for sbj in subjects:
                idx = [j for j, (s, r) in enumerate(zip(summary['subjects'], summary['rois']))
                       if s == sbj and r == roi]
                if idx:
                    subj_means.append(np.mean([summary['mean_corr'][j] for j in idx]))
                else:
                    subj_means.append(0)

            ax.bar(x + i * width, subj_means, width, label=roi)

        ax.set_xlabel('Subject', fontsize=12)
        ax.set_ylabel('Decoding Accuracy (Correlation)', fontsize=12)
        ax.set_title('Decoding Accuracy by Subject and ROI', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(subjects)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {save_path}")

        plt.show()

    def plot_text_similarity(self, summary, save_path=None):
        """Plot text generation similarity scores

        Parameters:
        -----------
        summary : dict
            Summary with text results
        save_path : str, optional
            Path to save figure
        """
        if not summary['subjects']:
            print("No text generation results to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        subjects = sorted(set(summary['subjects']))
        rois = sorted(set(summary['rois']))

        x = np.arange(len(subjects))
        width = 0.35

        for i, roi in enumerate(rois):
            subj_means = []
            for sbj in subjects:
                idx = [j for j, (s, r) in enumerate(zip(summary['subjects'], summary['rois']))
                       if s == sbj and r == roi]
                if idx:
                    subj_means.append(np.mean([summary['scores'][j] for j in idx]))
                else:
                    subj_means.append(0)

            ax.bar(x + i * width, subj_means, width, label=roi)

        ax.set_xlabel('Subject', fontsize=12)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title('Text Generation Similarity by Subject and ROI', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(subjects)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {save_path}")

        plt.show()

    def print_generated_examples(self, summary, n_examples=5):
        """Print examples of generated text

        Parameters:
        -----------
        summary : dict
            Summary with text results
        n_examples : int
            Number of examples to print
        """
        print("\n" + "="*60)
        print("Example Generated Texts")
        print("="*60)

        if not summary['generated_text']:
            print("No generated texts found")
            return

        # Print random examples
        indices = np.random.choice(len(summary['generated_text']),
                                   min(n_examples, len(summary['generated_text'])),
                                   replace=False)

        for idx in indices:
            print(f"\nSubject: {summary['subjects'][idx]}, ROI: {summary['rois'][idx]}")
            print(f"Sample: {summary['sample_idx'][idx]}")
            print(f"Generated: {summary['generated_text'][idx]}")
            print(f"Score: {summary['scores'][idx]:.4f}")

        print("="*60)

    def generate_all_figures(self, data_type='testPerception'):
        """Generate all summary figures

        Parameters:
        -----------
        data_type : str
            Data type to summarize
        """
        print("\nLoading decoding results...")
        dec_results = self.load_decoding_results(data_type)
        dec_summary = self.summarize_decoding_accuracy(dec_results)

        if dec_summary['subjects']:
            print("Generating decoding accuracy figure...")
            save_path1 = os.path.join(self.fig_dir, f'{data_type}_decoding_accuracy.png')
            self.plot_decoding_accuracy(dec_summary, save_path1)

        print("\nLoading text generation results...")
        txt_results = self.load_text_generation_results(data_type)
        txt_summary = self.summarize_text_generation(txt_results)

        if txt_summary['subjects']:
            print("Generating text similarity figure...")
            save_path2 = os.path.join(self.fig_dir, f'{data_type}_text_similarity.png')
            self.plot_text_similarity(txt_summary, save_path2)

            print("\nPrinting generated text examples...")
            self.print_generated_examples(txt_summary, n_examples=5)

        print("\nAll figures generated successfully!")


def main():
    """Main function"""
    # Initialize summary
    summary = DecodingSummary(root_path='./')

    # Generate figures for test perception
    summary.generate_all_figures('testPerception')

    # Generate figures for test imagery if available
    summary.generate_all_figures('testImagery')

    print("\nEnd decoding summary")


if __name__ == "__main__":
    main()
