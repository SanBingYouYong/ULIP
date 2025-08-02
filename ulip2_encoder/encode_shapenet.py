#!/usr/bin/env python3
"""
Encode ShapeNet or other .npy pointcloud files using ULIP2 encoder.

This script processes directories containing .npy pointcloud files and generates
embeddings using the ULIP2 model. It's specifically designed for batch processing
of pointcloud datasets.

Usage:
    python encode_shapenet.py --input_dir /path/to/npy/files --output_dir /path/to/embeddings
    python encode_shapenet.py --input_dir shapenet_pc/ --output_dir embeddings/ --batch_size 32
    python encode_shapenet.py --input_dir data/ --output_dir embeddings/ --device cpu
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict

# Add current directory to path to import ulip2_encoder
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from ulip2_encoder import ULIP2Encoder


def find_npy_files(input_dir: str) -> List[Path]:
    """Find all .npy files in the input directory recursively."""
    input_path = Path(input_dir)
    npy_files = list(input_path.glob("**/*.npy"))
    return npy_files


def process_pointcloud_batch(encoder: ULIP2Encoder, files: List[Path], 
                           input_dir: Path, output_dir: Path) -> Dict:
    """
    Process a batch of pointcloud files and save embeddings.
    
    Args:
        encoder: ULIP2Encoder instance
        files: List of .npy file paths to process
        input_dir: Input directory path
        output_dir: Output directory path
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {'success': 0, 'failed': 0, 'failed_files': []}
    
    for file_path in tqdm(files, desc="Encoding pointclouds"):
        try:
            # Generate embedding
            embedding = encoder.encode_pointcloud_file(str(file_path))
            
            # Create output path maintaining directory structure
            relative_path = file_path.relative_to(input_dir)
            output_path = output_dir / f"{relative_path.stem}_embedding.npy"
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save embedding
            np.save(output_path, embedding)
            stats['success'] += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            stats['failed'] += 1
            stats['failed_files'].append(str(file_path))
    
    return stats


def save_metadata(output_dir: Path, stats: Dict, args: argparse.Namespace):
    """Save processing metadata and statistics."""
    metadata = {
        'input_directory': str(args.input_dir),
        'output_directory': str(args.output_dir),
        'model_path': args.model_path,
        'device': args.device,
        'num_points': args.num_points,
        'total_files_processed': stats['success'] + stats['failed'],
        'successful_encodings': stats['success'],
        'failed_encodings': stats['failed'],
        'failed_files': stats['failed_files']
    }
    
    metadata_path = output_dir / 'encoding_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Encode .npy pointcloud files using ULIP2")
    
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing .npy pointcloud files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save embeddings')
    
    # Model arguments
    parser.add_argument('--model_path', type=str,
                       default='ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt',
                       help='Path to ULIP2 pretrained model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--num_points', type=int, default=10000,
                       help='Number of points to sample from pointclouds')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Number of files to process in each batch (for memory management)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file {model_path} does not exist")
        print("Please ensure the ULIP2 model is downloaded and the path is correct")
        return 1
    
    # Find all .npy files
    print(f"Searching for .npy files in {input_dir}")
    npy_files = find_npy_files(str(input_dir))
    
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return 1
    
    print(f"Found {len(npy_files)} .npy files")
    
    # Limit files if specified
    if args.max_files:
        npy_files = npy_files[:args.max_files]
        print(f"Processing first {len(npy_files)} files")
    
    # Initialize encoder
    print(f"Initializing ULIP2 encoder with model: {model_path}")
    try:
        encoder = ULIP2Encoder(str(model_path), args.device, args.num_points)
    except Exception as e:
        print(f"Error initializing encoder: {str(e)}")
        return 1
    
    # Process files in batches
    total_stats = {'success': 0, 'failed': 0, 'failed_files': []}
    
    # Split files into batches
    for i in range(0, len(npy_files), args.batch_size):
        batch_files = npy_files[i:i + args.batch_size]
        print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(npy_files) + args.batch_size - 1)//args.batch_size}")
        
        batch_stats = process_pointcloud_batch(encoder, batch_files, input_dir, output_dir)
        
        # Update total statistics
        total_stats['success'] += batch_stats['success']
        total_stats['failed'] += batch_stats['failed']
        total_stats['failed_files'].extend(batch_stats['failed_files'])
        
        # Clear GPU cache if using CUDA
        if args.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save metadata
    save_metadata(output_dir, total_stats, args)
    
    # Print summary
    print(f"\n=== Encoding Complete ===")
    print(f"Total files processed: {total_stats['success'] + total_stats['failed']}")
    print(f"Successful encodings: {total_stats['success']}")
    print(f"Failed encodings: {total_stats['failed']}")
    
    if total_stats['failed'] > 0:
        print(f"Failed files:")
        for failed_file in total_stats['failed_files'][:10]:  # Show first 10
            print(f"  - {failed_file}")
        if len(total_stats['failed_files']) > 10:
            print(f"  ... and {len(total_stats['failed_files']) - 10} more")
    
    print(f"Embeddings saved to: {output_dir}")
    
    return 0 if total_stats['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())
