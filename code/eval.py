#!/usr/bin/env python
"""
GP-Sparse Attention Evaluation Script

Evaluates trained models and compares sparse vs dense attention.

Usage:
    python eval.py --checkpoint_dense checkpoints/model_dense_best.pt --checkpoint_sparse checkpoints/model_sparse_best.pt

Note: Requires installing the renderformer package first:
    cd code && pip install -e .
"""

import sys
from pathlib import Path
import argparse
import json
import torch
import numpy as np
import time
from tqdm import tqdm
import logging

from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_dict = checkpoint.get('config', {})
    config = RenderFormerConfig(**config_dict)

    model = RenderFormer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    return model, config


def benchmark_inference(model, device, num_iterations: int = 5, num_views: int = 2) -> dict:
    """Benchmark inference speed and memory."""
    model.eval()

    times = []
    memory_usage = []

    with torch.no_grad():
        for _ in range(num_iterations):
            # Create mock batch
            batch_size, num_tri = 1, 1024
            tri_vpos = torch.randn(batch_size, num_tri, 9, device=device)
            texture_patches = torch.randn(batch_size, num_tri, 13, 32, 32, device=device)
            valid_mask = torch.ones(batch_size, num_tri, dtype=torch.bool, device=device)
            vns = torch.randn(batch_size, num_tri, 9, device=device)
            rays_o = torch.randn(batch_size, num_views, 3, device=device)
            rays_d = torch.randn(batch_size, num_views, 256, 256, 3, device=device)
            tri_vpos_view = tri_vpos.unsqueeze(1).expand(-1, num_views, -1, -1)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            _ = model(tri_vpos, texture_patches, valid_mask, vns, rays_o, rays_d, tri_vpos_view)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

            times.append(t1 - t0)

            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.max_memory_allocated() / 1e9)

    return {
        'time_mean_ms': np.mean(times) * 1000,
        'time_std_ms': np.std(times) * 1000,
        'time_min_ms': np.min(times) * 1000,
        'time_max_ms': np.max(times) * 1000,
        'memory_gb': np.mean(memory_usage) if memory_usage else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate GP-Sparse Attention models')
    parser.add_argument('--checkpoint_sparse', type=Path, help='Path to sparse model checkpoint')
    parser.add_argument('--checkpoint_dense', type=Path, help='Path to dense model checkpoint')
    parser.add_argument('--output_dir', type=Path, default=Path('eval_results'), help='Output directory for results')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of benchmark iterations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)

    results = {}

    # Benchmark sparse model
    if args.checkpoint_sparse:
        logger.info(f"Loading sparse model from {args.checkpoint_sparse}")
        model_sparse, config_sparse = load_model(args.checkpoint_sparse, device)

        logger.info("Benchmarking sparse model...")
        sparse_results = benchmark_inference(model_sparse, device, num_iterations=args.num_iterations)
        results['sparse'] = sparse_results
        logger.info(f"Sparse model: {sparse_results['time_mean_ms']:.2f} ± {sparse_results['time_std_ms']:.2f} ms")

    # Benchmark dense model
    if args.checkpoint_dense:
        logger.info(f"Loading dense model from {args.checkpoint_dense}")
        model_dense, config_dense = load_model(args.checkpoint_dense, device)

        logger.info("Benchmarking dense model...")
        dense_results = benchmark_inference(model_dense, device, num_iterations=args.num_iterations)
        results['dense'] = dense_results
        logger.info(f"Dense model: {dense_results['time_mean_ms']:.2f} ± {dense_results['time_std_ms']:.2f} ms")

    # Comparison
    if 'sparse' in results and 'dense' in results:
        speedup = results['dense']['time_mean_ms'] / results['sparse']['time_mean_ms']
        memory_reduction = results['dense']['memory_gb'] / max(results['sparse']['memory_gb'], 1e-6)

        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARISON: Dense vs Sparse")
        logger.info(f"{'='*60}")
        logger.info(f"Speedup: {speedup:.2f}×")
        logger.info(f"Memory reduction: {memory_reduction:.2f}×")
        logger.info(f"Dense time:  {results['dense']['time_mean_ms']:.2f} ms")
        logger.info(f"Sparse time: {results['sparse']['time_mean_ms']:.2f} ms")

        results['comparison'] = {
            'speedup': speedup,
            'memory_reduction': memory_reduction,
        }

    # Save results
    results_path = args.output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
