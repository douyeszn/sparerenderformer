#!/usr/bin/env python
"""
GP-Sparse Attention Training Script

Usage:
    python train.py --dataset ./training_data --num_epochs 100 --use_sparse
    python train.py --dataset ./training_data --num_epochs 10 --baseline  # Dense baseline

Note: Requires installing the renderformer package first:
    cd code && pip install -e .
"""

import sys
from pathlib import Path
import argparse
import dataclasses
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Tuple

from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(data_dir: Path, batch_size: int = 2):
    """Load training dataset from HDF5 files."""
    import h5py
    import torch.utils.data as data_utils

    hdf5_files = sorted(data_dir.glob('*.h5'))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

    logger.info(f"Found {len(hdf5_files)} HDF5 files")

    class RenderFormerDataset(torch.utils.data.Dataset):
        def __init__(self, hdf5_files):
            self.hdf5_files = hdf5_files
            self.cache = {}

        def __len__(self):
            return len(self.hdf5_files)

        def __getitem__(self, idx):
            if idx in self.cache:
                return self.cache[idx]

            with h5py.File(self.hdf5_files[idx], 'r') as f:
                vertices = torch.tensor(f['vertices'][:], dtype=torch.float32)
                faces = torch.tensor(f['triangles'][:], dtype=torch.long)
                images = torch.tensor(f['images'][:], dtype=torch.float32)

                v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
                face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
                face_normals = torch.nn.functional.normalize(face_normals, dim=1)

                tri_vpos = torch.cat([v0.reshape(-1, 3), v1.reshape(-1, 3), v2.reshape(-1, 3)], dim=1)
                vns = face_normals.repeat_interleave(3, dim=0).reshape(-1, 3, 3).transpose(1, 2).reshape(-1, 9)

                item = {'tri_vpos': tri_vpos, 'vns': vns, 'images': images}
                self.cache[idx] = item
                return item

    def collate_fn(batch):
        max_triangles = max(item['tri_vpos'].shape[0] for item in batch)
        padded_batch = {'tri_vpos': [], 'vns': [], 'images': [], 'valid_mask': []}

        for item in batch:
            n_tri = item['tri_vpos'].shape[0]
            pad_amount = max_triangles - n_tri

            padded_batch['tri_vpos'].append(torch.nn.functional.pad(item['tri_vpos'], (0, 0, 0, pad_amount)))
            padded_batch['vns'].append(torch.nn.functional.pad(item['vns'], (0, 0, 0, pad_amount)))

            mask = torch.ones(max_triangles, dtype=torch.bool)
            mask[n_tri:] = False
            padded_batch['valid_mask'].append(mask)
            padded_batch['images'].append(item['images'])

        for key in ['tri_vpos', 'vns', 'valid_mask']:
            padded_batch[key] = torch.stack(padded_batch[key])
        padded_batch['images'] = torch.cat(padded_batch['images'], dim=0)

        return padded_batch

    dataset = RenderFormerDataset(hdf5_files)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    return dataloader


def train_epoch(model, dataloader, device, optimizer, epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)):
        tri_vpos = batch['tri_vpos'].to(device)
        vns = batch['vns'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        batch_size, num_tri, _ = tri_vpos.shape
        num_views = 2

        # Create mock data for forward pass
        texture_patches = torch.randn(batch_size, num_tri, 13, 32, 32, device=device)
        rays_o = torch.randn(batch_size, num_views, 3, device=device) * 0.1 + 1.5
        rays_d = torch.randn(batch_size, num_views, 256, 256, 3, device=device)
        tri_vpos_view = tri_vpos.unsqueeze(1).expand(-1, num_views, -1, -1)

        optimizer.zero_grad()

        try:
            output = model(tri_vpos, texture_patches, valid_mask, vns, rays_o, rays_d, tri_vpos_view)
            # Use real rendered images from HDF5 as training targets
            images = batch['images'].to(device)  # (batch_size * hdf5_views, H, W, 3)
            hdf5_views = images.shape[0] // batch_size
            # Reshape to (batch_size, hdf5_views, H, W, 3) and take the first num_views
            target = images.view(batch_size, hdf5_views, *images.shape[1:])[:, :num_views]
            loss = nn.functional.mse_loss(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train GP-Sparse Attention model')
    parser.add_argument('--dataset', type=Path, default=Path('training_data'), help='Path to training data directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use_sparse', action='store_true', help='Use sparse attention (default: dense)')
    parser.add_argument('--baseline', action='store_true', help='Train dense baseline')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path('checkpoints'), help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

    args = parser.parse_args()
    args.checkpoint_dir.mkdir(exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.dataset}")
    train_loader = load_dataset(args.dataset, batch_size=args.batch_size)

    # Create model
    device = torch.device(args.device)
    use_sparse = args.use_sparse and not args.baseline

    config = RenderFormerConfig(
        latent_dim=768,
        num_layers=12,
        num_heads=6,
        dim_feedforward=3072,
        use_sparse_attention=use_sparse,
        sparse_k_local=64 if use_sparse else None,
        sparse_use_normal_mask=True if use_sparse else False,
    )

    model = RenderFormer(config).to(device)
    model_name = 'sparse' if use_sparse else 'dense'
    logger.info(f"Initialized {model_name} model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    losses = []
    best_loss = float('inf')

    logger.info(f"Starting training ({args.num_epochs} epochs)...")
    for epoch in range(args.num_epochs):
        loss = train_epoch(model, train_loader, device, optimizer, epoch)
        losses.append(loss)
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{args.num_epochs}: loss={loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            checkpoint_path = args.checkpoint_dir / f'model_{model_name}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': dataclasses.asdict(config),
            }, checkpoint_path)
            logger.info(f"Saved best checkpoint: {checkpoint_path}")

    # Final checkpoint
    final_path = args.checkpoint_dir / f'model_{model_name}_final.pt'
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'config': dataclasses.asdict(config),
        'losses': losses,
    }, final_path)

    # Save summary
    summary = {
        'model': model_name,
        'num_epochs': args.num_epochs,
        'final_loss': losses[-1],
        'best_loss': best_loss,
        'num_params': sum(p.numel() for p in model.parameters()),
    }

    summary_path = args.checkpoint_dir / f'summary_{model_name}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training complete! Final loss: {losses[-1]:.4f}")
    logger.info(f"Checkpoints saved to {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
