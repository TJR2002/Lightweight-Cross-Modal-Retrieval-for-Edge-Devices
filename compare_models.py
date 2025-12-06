#!/usr/bin/env python3
"""
Model Comparison and Benchmarking Script
Compare performance across all compression modes: none, distilled, pruned, quantized
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import argparse

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from cmrs_tricompression import (
    CrossModalRetrievalSystem,
    CompressionMode,
    MediaProcessor,
    config
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark different compression modes"""
    
    def __init__(self, test_data_dir: str):
        self.test_data_dir = Path(test_data_dir)
        self.processor = MediaProcessor()
        self.results = {}
        
    def collect_test_data(self, max_samples: int = 1000):
        """Collect test images and queries"""
        logger.info(f"Collecting test data from {self.test_data_dir}")
        
        # Collect images
        image_paths = []
        for ext in config.SUPPORTED_IMAGE_FORMATS:
            image_paths.extend(list(self.test_data_dir.rglob(f"*{ext}")))
        
        image_paths = image_paths[:max_samples]
        
        if not image_paths:
            raise ValueError(f"No images found in {self.test_data_dir}")
        
        logger.info(f"Found {len(image_paths)} test images")
        
        # Generate test queries
        queries = [
            "person",
            "animal",
            "car",
            "building",
            "nature",
            "food",
            "indoor scene",
            "outdoor scene",
            "sunset",
            "beach"
        ]
        
        return [str(p) for p in image_paths], queries
    
    def benchmark_mode(self, mode: CompressionMode, image_paths: List[str], queries: List[str]):
        """Benchmark a single compression mode"""
        logger.info(f"\nBenchmarking {mode.value} mode...")
        
        try:
            # Initialize system
            system = CrossModalRetrievalSystem(compression_mode=mode)
            
            # Measure model size
            if hasattr(system.model, 'get_model_size_mb'):
                model_size_mb = system.model.get_model_size_mb()
            else:
                # For CLIP model
                model_size_mb = sum(
                    p.numel() * p.element_size() for p in system.model.parameters()
                ) / (1024 ** 2)
            
            # Count parameters
            if hasattr(system.model, 'count_parameters'):
                params = system.model.count_parameters()
            else:
                params = sum(p.numel() for p in system.model.parameters())
            
            # Benchmark encoding speed
            encoding_times = []
            logger.info("Benchmarking encoding speed...")
            
            for img_path in tqdm(image_paths[:100], desc="Encoding images"):
                start = time.time()
                _ = system.encode_media(img_path)
                encoding_times.append(time.time() - start)
            
            avg_encoding_time = np.mean(encoding_times) * 1000  # Convert to ms
            
            # Index all images
            logger.info("Indexing images...")
            for img_path in tqdm(image_paths, desc="Building index"):
                embedding = system.encode_media(img_path)
                if embedding is not None:
                    # Add to index (simplified, doesn't use actual add_media_directory)
                    pass
            
            # Benchmark search speed
            search_times = []
            logger.info("Benchmarking search speed...")
            
            for query in tqdm(queries, desc="Searching"):
                start = time.time()
                _ = system.encode_text(query)
                search_times.append(time.time() - start)
            
            avg_search_time = np.mean(search_times) * 1000  # Convert to ms
            
            # Memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 ** 2)
            
            results = {
                'mode': mode.value,
                'parameters': params,
                'model_size_mb': model_size_mb,
                'avg_encoding_time_ms': avg_encoding_time,
                'avg_search_time_ms': avg_search_time,
                'memory_usage_mb': memory_mb,
                'total_images': len(image_paths)
            }
            
            logger.info(f"\nResults for {mode.value}:")
            logger.info(f"  Parameters: {params:,}")
            logger.info(f"  Model size: {model_size_mb:.2f} MB")
            logger.info(f"  Encoding: {avg_encoding_time:.2f} ms/image")
            logger.info(f"  Search: {avg_search_time:.2f} ms/query")
            logger.info(f"  Memory: {memory_mb:.1f} MB")
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking {mode.value}: {e}")
            return None
    
    def run_full_benchmark(self, modes: List[CompressionMode] = None):
        """Run benchmark on all specified modes"""
        if modes is None:
            modes = [
                CompressionMode.NONE,
                CompressionMode.DISTILLED,
                CompressionMode.PRUNED,
                CompressionMode.QUANTIZED
            ]
        
        # Collect test data
        image_paths, queries = self.collect_test_data()
        
        # Run benchmarks
        for mode in modes:
            result = self.benchmark_mode(mode, image_paths, queries)
            if result:
                self.results[mode.value] = result
        
        return self.results
    
    def save_results(self, output_path: str = "benchmark_results.json"):
        """Save benchmark results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    def generate_comparison_plots(self, output_dir: str = "benchmark_plots"):
        """Generate comparison plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Prepare data
        modes = list(self.results.keys())
        params = [self.results[m]['parameters'] / 1e6 for m in modes]  # In millions
        sizes = [self.results[m]['model_size_mb'] for m in modes]
        encoding_times = [self.results[m]['avg_encoding_time_ms'] for m in modes]
        search_times = [self.results[m]['avg_search_time_ms'] for m in modes]
        memory = [self.results[m]['memory_usage_mb'] for m in modes]
        
        # Set style
        sns.set_style("whitegrid")
        
        # Plot 1: Model Size Comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modes, sizes, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        plt.xlabel('Compression Mode', fontsize=12)
        plt.ylabel('Model Size (MB)', fontsize=12)
        plt.title('Model Size Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size:.1f} MB',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Parameters Comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modes, params, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        plt.xlabel('Compression Mode', fontsize=12)
        plt.ylabel('Parameters (Millions)', fontsize=12)
        plt.title('Model Parameters Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        for bar, param in zip(bars, params):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{param:.1f}M',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/parameters.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Speed Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Encoding speed
        bars1 = ax1.bar(modes, encoding_times, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        ax1.set_xlabel('Compression Mode', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Encoding Speed', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars1, encoding_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f} ms',
                    ha='center', va='bottom', fontsize=9)
        
        # Search speed
        bars2 = ax2.bar(modes, search_times, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        ax2.set_xlabel('Compression Mode', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('Search Speed', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars2, search_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f} ms',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/speed_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Memory Usage
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modes, memory, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        plt.xlabel('Compression Mode', fontsize=12)
        plt.ylabel('Memory Usage (MB)', fontsize=12)
        plt.title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        for bar, mem in zip(bars, memory):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem:.0f} MB',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 5: Compression vs Performance Trade-off
        plt.figure(figsize=(10, 6))
        
        # Normalize to baseline (CLIP)
        baseline_size = sizes[0]
        normalized_sizes = [s / baseline_size for s in sizes]
        
        # Calculate compression ratios
        compression_ratios = [baseline_size / s for s in sizes]
        
        # Plot compression vs speed
        plt.scatter(compression_ratios, encoding_times, s=200, 
                   c=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], alpha=0.6)
        
        for i, mode in enumerate(modes):
            plt.annotate(mode, (compression_ratios[i], encoding_times[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.xlabel('Compression Ratio (vs CLIP)', fontsize=12)
        plt.ylabel('Encoding Time (ms)', fontsize=12)
        plt.title('Compression vs Performance Trade-off', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {output_dir}/")
    
    def print_summary_table(self):
        """Print summary table"""
        if not self.results:
            logger.warning("No results to display")
            return
        
        print("\n" + "="*100)
        print("BENCHMARK SUMMARY")
        print("="*100)
        
        # Header
        print(f"{'Mode':<12} {'Params':<15} {'Size (MB)':<12} "
              f"{'Encoding (ms)':<15} {'Search (ms)':<15} {'Memory (MB)':<12}")
        print("-"*100)
        
        # Data rows
        for mode, result in self.results.items():
            print(f"{mode:<12} "
                  f"{result['parameters']/1e6:>8.1f}M     "
                  f"{result['model_size_mb']:>8.1f}     "
                  f"{result['avg_encoding_time_ms']:>10.2f}       "
                  f"{result['avg_search_time_ms']:>10.2f}       "
                  f"{result['memory_usage_mb']:>8.0f}")
        
        print("="*100)
        
        # Compression ratios
        if 'none' in self.results:
            baseline = self.results['none']
            print("\nCompression Ratios (vs CLIP baseline):")
            print("-"*100)
            
            for mode, result in self.results.items():
                if mode != 'none':
                    size_ratio = baseline['model_size_mb'] / result['model_size_mb']
                    param_ratio = baseline['parameters'] / result['parameters']
                    speed_improvement = baseline['avg_encoding_time_ms'] / result['avg_encoding_time_ms']
                    
                    print(f"{mode:<12} "
                          f"Size: {size_ratio:>5.1f}x  "
                          f"Params: {param_ratio:>5.1f}x  "
                          f"Speed: {speed_improvement:>5.2f}x faster")
            
            print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tri-compression model performance"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with test images"
    )
    
    parser.add_argument(
        "--modes",
        nargs='+',
        choices=['none', 'distilled', 'pruned', 'quantized'],
        default=None,
        help="Modes to benchmark (default: all)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for results"
    )
    
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="benchmark_plots",
        help="Directory for output plots"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    args = parser.parse_args()
    
    # Convert mode strings to enum
    if args.modes:
        modes = [CompressionMode(m) for m in args.modes]
    else:
        modes = None
    
    # Run benchmark
    benchmark = ModelBenchmark(args.data_dir)
    
    print("\n" + "="*100)
    print("TRI-COMPRESSION MODEL BENCHMARK")
    print("="*100)
    
    benchmark.run_full_benchmark(modes)
    
    # Save results
    benchmark.save_results(args.output)
    
    # Print summary
    benchmark.print_summary_table()
    
    # Generate plots
    if not args.no_plots:
        try:
            benchmark.generate_comparison_plots(args.plot_dir)
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
            logger.warning("Install matplotlib and seaborn for plot generation:")
            logger.warning("  pip install matplotlib seaborn")
    
    print(f"\n✓ Benchmark complete! Results saved to {args.output}")
    if not args.no_plots:
        print(f"✓ Plots saved to {args.plot_dir}/")


if __name__ == "__main__":
    main()