#!/usr/bin/env python
"""
Convert 3D models to RenderFormer HDF5 format.

Usage:
    python convert_to_hdf5.py --input_dir ./objaverse_models --output_dir ./training_data
"""

import argparse
import h5py
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional


def load_mesh(filepath: str) -> Optional[trimesh.Trimesh]:
    """Load mesh from file. Tries multiple formats."""
    try:
        mesh = trimesh.load(filepath, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(merged=True)
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        mesh.remove_degenerate_faces()
        mesh.merge_vertices()
        return mesh
    except Exception:
        return None


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to [-0.5, 0.5] (RenderFormer training range)."""
    mesh.vertices -= mesh.center_mass
    scale = mesh.extents.max()
    if scale > 0:
        mesh.vertices /= (scale / 0.4)
    return mesh


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms
    return normals


def create_random_materials(num_triangles: int, seed: int = None) -> Dict[str, np.ndarray]:
    """Generate random PBR materials."""
    if seed is not None:
        np.random.seed(seed)
    return {
        'diffuse_albedo': np.random.rand(num_triangles, 3) * 0.8 + 0.1,
        'specular_albedo': np.random.rand(num_triangles, 3) * 0.5 + 0.2,
        'roughness': np.random.rand(num_triangles) * 0.8 + 0.1,
        'metallic': np.random.rand(num_triangles) * 0.5,
    }


def create_training_hdf5(
    mesh: trimesh.Trimesh,
    output_path: str,
    seed: int,
    num_views: int = 4
) -> str:
    """Create HDF5 file in RenderFormer format from a mesh."""
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    face_normals = compute_face_normals(vertices, faces).astype(np.float32)
    materials = create_random_materials(len(faces), seed=seed)

    # Generate camera poses
    camera_poses = []
    for i in range(num_views):
        angle = (i / num_views) * 2 * np.pi
        radius = 1.8
        height = 0.2
        cam_pos = np.array(
            [radius * np.cos(angle), height, radius * np.sin(angle)],
            dtype=np.float32
        )
        forward = -cam_pos / np.linalg.norm(cam_pos)
        camera_poses.append({
            'position': cam_pos,
            'forward': forward,
            'up': np.array([0, 1, 0], dtype=np.float32),
            'fov': 45.0
        })

    # Synthetic images
    img_size = 256
    images = (np.random.rand(num_views, img_size, img_size, 3) * 0.5 + 0.25).astype(np.float32)

    # Write HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('vertices', data=vertices)
        f.create_dataset('triangles', data=faces)
        f.create_dataset('vertex_normals', data=mesh.vertex_normals.astype(np.float32))
        f.create_dataset('diffuse_albedo', data=materials['diffuse_albedo'])
        f.create_dataset('specular_albedo', data=materials['specular_albedo'])
        f.create_dataset('roughness', data=materials['roughness'].astype(np.float32))
        f.create_dataset('metallic', data=materials['metallic'].astype(np.float32))
        f.create_dataset('images', data=images)
        camera_group = f.create_group('cameras')
        for i, cam in enumerate(camera_poses):
            cam_subgroup = camera_group.create_group(f'camera_{i}')
            for key, val in cam.items():
                if isinstance(val, np.ndarray):
                    cam_subgroup.create_dataset(key, data=val)
                else:
                    cam_subgroup.attrs[key] = val
    return output_path


def convert_models_to_hdf5(
    input_dir: Path,
    output_dir: Path,
    max_faces: int = 5000,
    min_faces: int = 100,
    num_views: int = 4,
    max_models: Optional[int] = None,
) -> list:
    """
    Convert all models in input_dir to HDF5 files.

    Args:
        input_dir: Directory containing 3D models
        output_dir: Directory to save HDF5 files
        max_faces: Maximum faces for decimation
        min_faces: Minimum faces to include model
        num_views: Number of camera views per model
        max_models: Maximum models to process (None = all)

    Returns:
        List of created HDF5 file paths
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all 3D model files
    model_files = []
    for ext in ['*.obj', '*.glb', '*.gltf', '*.ply', '*.stl', '*.fbx']:
        model_files.extend(input_dir.glob(f'**/{ext}'))

    if max_models:
        model_files = model_files[:max_models]

    hdf5_files = []
    print(f"Converting {len(model_files)} models to HDF5...\n")

    for idx, model_path in enumerate(tqdm(model_files, desc="Converting models")):
        try:
            mesh = load_mesh(str(model_path))
            if mesh is None or len(mesh.vertices) < 3 or len(mesh.faces) < min_faces:
                continue

            # Simplify if too many faces
            if len(mesh.faces) > max_faces:
                target_count = int(np.random.randint(min_faces, max_faces))
                mesh = mesh.simplify_quadratic_decimation(target_count=target_count)

            # Skip if still too small
            if len(mesh.faces) < min_faces:
                continue

            mesh = normalize_mesh(mesh)
            output_file = output_dir / f'scene_{idx:04d}_{model_path.stem[:8]}.h5'
            create_training_hdf5(mesh, str(output_file), seed=idx, num_views=num_views)
            hdf5_files.append(output_file)
        except Exception:
            continue

    print(f"\n✓ Created {len(hdf5_files)} HDF5 files in {output_dir}")
    return hdf5_files


def main():
    parser = argparse.ArgumentParser(description='Convert 3D models to RenderFormer HDF5 format')
    parser.add_argument('--input_dir', type=Path, required=True, help='Input directory with 3D models')
    parser.add_argument('--output_dir', type=Path, default=Path('training_data'), help='Output directory for HDF5 files')
    parser.add_argument('--max_faces', type=int, default=5000, help='Maximum faces for decimation')
    parser.add_argument('--min_faces', type=int, default=100, help='Minimum faces to include model')
    parser.add_argument('--num_views', type=int, default=4, help='Number of camera views per model')
    parser.add_argument('--max_models', type=int, default=None, help='Maximum models to process')

    args = parser.parse_args()

    hdf5_files = convert_models_to_hdf5(
        args.input_dir,
        args.output_dir,
        max_faces=args.max_faces,
        min_faces=args.min_faces,
        num_views=args.num_views,
        max_models=args.max_models,
    )

    return len(hdf5_files)


if __name__ == '__main__':
    main()
