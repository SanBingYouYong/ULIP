"""
ULIP2 Encoder for 3D Model Retrieval

This module provides functionality to encode 3D models (.obj files or point clouds) 
and text descriptions using ULIP2 for similarity-based retrieval.

Usage examples:
1. Encode a single OBJ file:
   python ulip2_encoder.py --obj_file model.obj --output_dir embeddings/

2. Encode all OBJ files in a directory:
   python ulip2_encoder.py --obj_dir models/ --output_dir embeddings/

3. Encode text descriptions:
   python ulip2_encoder.py --text "a red chair" --output_dir embeddings/

4. Compare 3D model with text:
   python ulip2_encoder.py --obj_file model.obj --text "a chair" --compare

5. Build vector database from directory:
   python ulip2_encoder.py --obj_dir models/ --build_db --db_path vector_db.pkl
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import open3d as o3d
import json
from tqdm import tqdm
from collections import OrderedDict

# Optional import for color mapping
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using fallback for color mapping")

# Add parent directory to Python path to access ULIP modules (when run from parent dir)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import ULIP2 components
from utils.tokenizer import SimpleTokenizer
import models.ULIP_models as models
from utils import utils
from data.dataset_3d import pc_normalize


class OBJProcessor:
    """Handles processing of OBJ files to point clouds."""
    
    def __init__(self, num_points: int = 10000):
        self.num_points = num_points
    
    def _parse_mtl_file(self, obj_path: str) -> Optional[np.ndarray]:
        """
        Parse MTL file to extract material colors.
        
        Args:
            obj_path: Path to the OBJ file
            
        Returns:
            RGB color array if found, None otherwise
        """
        try:
            obj_dir = os.path.dirname(obj_path)
            
            # Read OBJ file to find MTL reference
            with open(obj_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('mtllib '):
                        mtl_file = line.split()[1]
                        mtl_path = os.path.join(obj_dir, mtl_file)
                        
                        if os.path.exists(mtl_path):
                            return self._extract_color_from_mtl(mtl_path)
            return None
            
        except Exception as e:
            print(f"Warning: Could not parse MTL file for {obj_path}: {e}")
            return None
    
    def _extract_color_from_mtl(self, mtl_path: str) -> Optional[np.ndarray]:
        """
        Extract color from MTL file.
        
        Args:
            mtl_path: Path to the MTL file
            
        Returns:
            RGB color array if found, None otherwise
        """
        try:
            with open(mtl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Look for diffuse color (Kd) or ambient color (Ka)
                    if line.startswith('Kd ') or line.startswith('Ka '):
                        color_values = line.split()[1:4]
                        if len(color_values) == 3:
                            color = np.array([float(v) for v in color_values])
                            # Ensure color is in [0, 1] range
                            color = np.clip(color, 0, 1)
                            return color
            return None
            
        except Exception as e:
            print(f"Warning: Could not extract color from MTL file {mtl_path}: {e}")
            return None
    
    def load_obj_to_pointcloud(self, obj_path: str) -> np.ndarray:
        """
        Load an OBJ file and convert it to a point cloud.
        Extracts RGB colors from materials (MTL files) and textures when available.
        
        Args:
            obj_path: Path to the OBJ file
            
        Returns:
            Point cloud as numpy array of shape (N, 6) with XYZRGB coordinates
        """
        try:
            # Load mesh using Open3D with material loading enabled
            mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
            
            if len(mesh.vertices) == 0:
                raise ValueError(f"No vertices found in {obj_path}")
            
            # Try to load materials and textures
            colors_extracted = False
            if len(mesh.triangles) > 0:
                # Check if mesh has materials/textures
                if len(mesh.textures) > 0 or len(mesh.triangle_material_ids) > 0:
                    try:
                        # Sample points from mesh surface with color information
                        point_cloud = mesh.sample_points_uniformly(
                            number_of_points=self.num_points,
                            use_triangle_normal=True
                        )
                        
                        # Try to get colors from texture coordinates if available
                        if len(mesh.textures) > 0 and len(mesh.triangle_uvs) > 0:
                            # This is a more complex case - we'd need to interpolate texture colors
                            # For now, let's try a simpler approach with vertex colors
                            pass
                            
                    except Exception as tex_e:
                        print(f"Warning: Could not extract texture colors from {obj_path}: {tex_e}")
                
                # Fallback: Sample points from mesh surface
                if not colors_extracted:
                    point_cloud = mesh.sample_points_uniformly(number_of_points=self.num_points)
            else:
                # If no faces, just use vertices
                vertices = np.asarray(mesh.vertices)
                if len(vertices) >= self.num_points:
                    # Randomly sample points
                    indices = np.random.choice(len(vertices), self.num_points, replace=False)
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(vertices[indices])
                else:
                    # Upsample by repeating points
                    repeat_factor = (self.num_points // len(vertices)) + 1
                    vertices_repeated = np.tile(vertices, (repeat_factor, 1))[:self.num_points]
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(vertices_repeated)
            
            # Convert to numpy array
            points = np.asarray(point_cloud.points)
            
            # Try to extract colors from various sources
            colors = None
            
            # Method 1: Check if sampled point cloud has colors
            if len(point_cloud.colors) > 0:
                colors = np.asarray(point_cloud.colors)
                colors_extracted = True
                print(f"Extracted colors from point cloud sampling for {obj_path}")
            
            # Method 2: Check if original mesh has vertex colors
            elif len(mesh.vertex_colors) > 0:
                # Map vertex colors to sampled points (approximate)
                mesh_colors = np.asarray(mesh.vertex_colors)
                if len(mesh_colors) == len(np.asarray(mesh.vertices)):
                    if SCIPY_AVAILABLE:
                        # Use nearest neighbor mapping
                        mesh_vertices = np.asarray(mesh.vertices)
                        tree = cKDTree(mesh_vertices)
                        _, indices = tree.query(points[:, :3])
                        colors = mesh_colors[indices]
                        colors_extracted = True
                        print(f"Mapped vertex colors to point cloud for {obj_path}")
                    else:
                        # Fallback: use average color or simple mapping
                        avg_color = np.mean(mesh_colors, axis=0)
                        colors = np.tile(avg_color, (len(points), 1))
                        colors_extracted = True
                        print(f"Used average vertex color for {obj_path} (scipy not available)")
            
            # Method 3: Try to extract material colors from MTL file
            elif os.path.exists(obj_path):
                mtl_color = self._parse_mtl_file(obj_path)
                if mtl_color is not None:
                    colors = np.tile(mtl_color, (len(points), 1))
                    colors_extracted = True
                    print(f"Extracted color from MTL file for {obj_path}")
            
            # Method 4: Try to extract material colors from Open3D materials
            if not colors_extracted and hasattr(mesh, 'materials') and len(mesh.materials) > 0:
                # Use dominant material color
                try:
                    # Get the first material's diffuse color
                    material = mesh.materials[0]
                    if hasattr(material, 'baseColor'):
                        base_color = material.baseColor[:3]  # RGB part
                        colors = np.tile(base_color, (len(points), 1))
                        colors_extracted = True
                        print(f"Used material base color for {obj_path}")
                    elif hasattr(material, 'diffuse'):
                        diffuse_color = material.diffuse[:3]  # RGB part
                        colors = np.tile(diffuse_color, (len(points), 1))
                        colors_extracted = True
                        print(f"Used material diffuse color for {obj_path}")
                except Exception as mat_e:
                    print(f"Warning: Could not extract material colors from {obj_path}: {mat_e}")
            
            # Add colors to points
            if colors_extracted and colors is not None and len(colors) == len(points):
                # Ensure colors are in [0, 1] range
                if colors.max() > 1.0:
                    colors = colors / 255.0
                points = np.concatenate([points, colors], axis=1)
            else:
                # Fallback: Use neutral gray colors
                dummy_rgb = np.full((len(points), 3), 0.5)
                points = np.concatenate([points, dummy_rgb], axis=1)
                if not colors_extracted:
                    print(f"No color information found in {obj_path}, using gray")
            
            # Normalize point cloud coordinates
            points[:, :3] = pc_normalize(points[:, :3])
            
            return points
            
        except Exception as e:
            print(f"Error loading {obj_path}: {str(e)}")
            # Return a default point cloud if loading fails
            default_points = np.random.randn(self.num_points, 3) * 0.1
            default_points = pc_normalize(default_points)
            # Add dummy RGB channels (0.5, 0.5, 0.5) for gray
            dummy_rgb = np.full((self.num_points, 3), 0.5)
            default_points = np.concatenate([default_points, dummy_rgb], axis=1)
            return default_points
    
    def load_pointcloud_file(self, pc_path: str) -> np.ndarray:
        """
        Load a point cloud from various formats (.ply, .pcd, .txt, .npy).
        
        Args:
            pc_path: Path to the point cloud file
            
        Returns:
            Point cloud as numpy array
        """
        ext = Path(pc_path).suffix.lower()
        
        try:
            if ext in ['.ply', '.pcd']:
                pcd = o3d.io.read_point_cloud(pc_path)
                points = np.asarray(pcd.points)
                if len(pcd.colors) > 0:
                    colors = np.asarray(pcd.colors)
                    points = np.concatenate([points, colors], axis=1)
                    
            elif ext == '.npy':
                points = np.load(pc_path)
                
            elif ext == '.txt':
                points = np.loadtxt(pc_path)
                
            else:
                raise ValueError(f"Unsupported point cloud format: {ext}")
            
            # Ensure we have the right number of points
            if len(points) > self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
            elif len(points) < self.num_points:
                # Upsample by repeating points
                repeat_factor = (self.num_points // len(points)) + 1
                points = np.tile(points, (repeat_factor, 1))[:self.num_points]
            
            # Normalize XYZ coordinates
            points[:, :3] = pc_normalize(points[:, :3])
            
            return points
            
        except Exception as e:
            print(f"Error loading point cloud {pc_path}: {str(e)}")
            # Return default point cloud
            default_points = np.random.randn(self.num_points, 3) * 0.1
            default_points = pc_normalize(default_points)
            # Add dummy RGB channels (0.5, 0.5, 0.5) for gray
            dummy_rgb = np.full((self.num_points, 3), 0.5)
            default_points = np.concatenate([default_points, dummy_rgb], axis=1)
            return default_points


class ULIP2Encoder:
    """ULIP2-based encoder for 3D models and text."""
    
    def __init__(self, model_path: str, device: str = 'cuda', num_points: int = 10000):
        """
        Initialize the ULIP2 encoder.
        
        Args:
            model_path: Path to the pretrained ULIP2 model
            device: Device to run the model on ('cuda' or 'cpu')
            num_points: Number of points to sample from 3D models
        """
        self.device = device
        self.num_points = num_points
        self.obj_processor = OBJProcessor(num_points)
        self.tokenizer = SimpleTokenizer()
        
        # Load the model
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str):
        """Load the ULIP2 model from checkpoint."""
        print(f"Loading ULIP2 model from {model_path}")
        
        # Create dummy args for model initialization
        args = argparse.Namespace()
        args.evaluate_3d_ulip2 = True
        args.npoints = self.num_points
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        
        # Initialize model
        model = getattr(models, 'ULIP2_PointBERT_Colored')(args=args)
        model.to(self.device)
        model.load_state_dict(state_dict, strict=False)
        
        print("ULIP2 model loaded successfully")
        return model
    
    def encode_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """
        Encode a point cloud to embedding vector.
        
        Args:
            points: Point cloud as numpy array of shape (N, 3) or (N, 6)
            
        Returns:
            Embedding vector as numpy array
        """
        # Ensure correct shape and type
        if points.shape[1] < 3:
            raise ValueError("Point cloud must have at least 3 coordinates (XYZ)")
        
        # Take only first 3 or 6 dimensions (XYZ or XYZRGB)
        if points.shape[1] >= 6:
            points = points[:, :6]  # XYZRGB
        else:
            # XYZ only - add dummy RGB channels (0.5, 0.5, 0.5) for gray
            points_xyz = points[:, :3]
            dummy_rgb = np.full((points_xyz.shape[0], 3), 0.5, dtype=points_xyz.dtype)
            points = np.concatenate([points_xyz, dummy_rgb], axis=1)
        
        # Convert to tensor
        pc_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode point cloud
            pc_embed = self.model.encode_pc(pc_tensor)
            pc_embed = F.normalize(pc_embed, dim=-1)
            
        return pc_embed.cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Text description
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize text
        text_tokens = self.tokenizer([text]).to(self.device)
        
        # Ensure correct tensor shape - add batch dimension if needed
        if len(text_tokens.shape) < 2:
            text_tokens = text_tokens[None, ...]
        
        with torch.no_grad():
            # Encode text
            text_embed = self.model.encode_text(text_tokens)
            text_embed = F.normalize(text_embed, dim=-1)
            
        return text_embed.cpu().numpy()
    
    def encode_obj_file(self, obj_path: str) -> np.ndarray:
        """
        Encode an OBJ file to embedding vector.
        
        Args:
            obj_path: Path to the OBJ file
            
        Returns:
            Embedding vector as numpy array
        """
        points = self.obj_processor.load_obj_to_pointcloud(obj_path)
        return self.encode_pointcloud(points)
    
    def encode_pointcloud_file(self, pc_path: str) -> np.ndarray:
        """
        Encode a point cloud file to embedding vector.
        
        Args:
            pc_path: Path to the point cloud file
            
        Returns:
            Embedding vector as numpy array
        """
        points = self.obj_processor.load_pointcloud_file(pc_path)
        return self.encode_pointcloud(points)
    
    def compute_similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embed1: First embedding vector
            embed2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        embed1_norm = embed1 / np.linalg.norm(embed1)
        embed2_norm = embed2 / np.linalg.norm(embed2)
        return float(np.dot(embed1_norm.flatten(), embed2_norm.flatten()))


class VectorDatabase:
    """Simple vector database for storing and retrieving 3D model embeddings."""
    
    def __init__(self):
        self.embeddings = []
        self.metadata = []
        self.index_to_file = {}
    
    def add_embedding(self, embedding: np.ndarray, file_path: str, metadata: Dict = None):
        """Add an embedding to the database."""
        index = len(self.embeddings)
        self.embeddings.append(embedding.flatten())
        self.index_to_file[index] = file_path
        meta = metadata or {}
        meta['file_path'] = file_path
        meta['index'] = index
        self.metadata.append(meta)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (file_path, similarity_score, metadata)
        """
        if len(self.embeddings) == 0:
            return []
        
        query_norm = query_embedding.flatten() / np.linalg.norm(query_embedding.flatten())
        similarities = []
        
        for i, embed in enumerate(self.embeddings):
            embed_norm = embed / np.linalg.norm(embed)
            sim = float(np.dot(query_norm, embed_norm))
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for i, (idx, sim) in enumerate(similarities[:top_k]):
            file_path = self.index_to_file[idx]
            metadata = self.metadata[idx]
            results.append((file_path, sim, metadata))
        
        return results
    
    def save(self, db_path: str):
        """Save the database to file."""
        data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'index_to_file': self.index_to_file
        }
        with open(db_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Vector database saved to {db_path}")
    
    def load(self, db_path: str):
        """Load the database from file."""
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        self.index_to_file = data['index_to_file']
        print(f"Vector database loaded from {db_path}")


def encode_directory(encoder: ULIP2Encoder, input_dir: str, output_dir: str, 
                    file_extensions: List[str] = ['.obj', '.ply', '.pcd', '.npy', '.txt']):
    """
    Encode all 3D files in a directory.
    
    Args:
        encoder: ULIP2Encoder instance
        input_dir: Directory containing 3D files
        output_dir: Directory to save embeddings
        file_extensions: List of file extensions to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all 3D files
    files = []
    for ext in file_extensions:
        files.extend(Path(input_dir).glob(f"**/*{ext}"))
    
    print(f"Found {len(files)} 3D files to encode")
    
    # Encode each file
    for file_path in tqdm(files, desc="Encoding 3D files"):
        try:
            # Determine encoding method based on extension
            if file_path.suffix.lower() == '.obj':
                embedding = encoder.encode_obj_file(str(file_path))
            else:
                embedding = encoder.encode_pointcloud_file(str(file_path))
            
            # Save embedding
            relative_path = file_path.relative_to(input_dir)
            output_path = Path(output_dir) / f"{relative_path.stem}.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embedding)
            
        except Exception as e:
            print(f"Error encoding {file_path}: {str(e)}")


def build_vector_database(encoder: ULIP2Encoder, input_dir: str, db_path: str,
                         file_extensions: List[str] = ['.obj', '.ply', '.pcd', '.npy', '.txt']) -> VectorDatabase:
    """
    Build a vector database from all 3D files in a directory.
    
    Args:
        encoder: ULIP2Encoder instance
        input_dir: Directory containing 3D files
        db_path: Path to save the vector database
        file_extensions: List of file extensions to process
        
    Returns:
        VectorDatabase instance
    """
    db = VectorDatabase()
    
    # Find all 3D files
    files = []
    for ext in file_extensions:
        files.extend(Path(input_dir).glob(f"**/*{ext}"))
    
    print(f"Building vector database from {len(files)} 3D files")
    
    # Encode each file and add to database
    for file_path in tqdm(files, desc="Building vector database"):
        try:
            # Determine encoding method based on extension
            if file_path.suffix.lower() == '.obj':
                embedding = encoder.encode_obj_file(str(file_path))
            else:
                embedding = encoder.encode_pointcloud_file(str(file_path))
            
            # Add to database
            metadata = {
                'filename': file_path.name,
                'extension': file_path.suffix.lower(),
                'relative_path': str(file_path.relative_to(input_dir))
            }
            db.add_embedding(embedding, str(file_path), metadata)
            
        except Exception as e:
            print(f"Error encoding {file_path}: {str(e)}")
    
    # Save database
    db.save(db_path)
    return db


def main():
    parser = argparse.ArgumentParser(description="ULIP2 3D Model and Text Encoder")
    
    # Model arguments
    parser.add_argument('--model_path', type=str, 
                       default='ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt',
                       help='Path to ULIP2 pretrained model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--num_points', type=int, default=10000,
                       help='Number of points to sample from 3D models')
    
    # Input arguments
    parser.add_argument('--obj_file', type=str, help='Single OBJ file to encode')
    parser.add_argument('--obj_dir', type=str, help='Directory containing OBJ/3D files')
    parser.add_argument('--pc_file', type=str, help='Single point cloud file to encode')
    parser.add_argument('--text', type=str, help='Text description to encode')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='embeddings',
                       help='Directory to save embeddings')
    parser.add_argument('--output_file', type=str, help='Specific output file for single embedding')
    
    # Vector database arguments
    parser.add_argument('--build_db', action='store_true',
                       help='Build vector database from directory')
    parser.add_argument('--db_path', type=str, default='vector_database.pkl',
                       help='Path to save/load vector database')
    parser.add_argument('--search_db', action='store_true',
                       help='Search in existing vector database')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top results to return in search')
    
    # Comparison arguments
    parser.add_argument('--compare', action='store_true',
                       help='Compare 3D model with text description')
    
    args = parser.parse_args()
    
    # Initialize encoder
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please download the ULIP2 model or specify the correct path")
        return
    
    encoder = ULIP2Encoder(args.model_path, args.device, args.num_points)
    
    # Handle different modes
    if args.build_db:
        if not args.obj_dir:
            print("Error: --obj_dir is required for building vector database")
            return
        db = build_vector_database(encoder, args.obj_dir, args.db_path)
        print(f"Vector database built with {len(db.embeddings)} embeddings")
        
    elif args.search_db:
        if not os.path.exists(args.db_path):
            print(f"Error: Vector database not found at {args.db_path}")
            return
        
        db = VectorDatabase()
        db.load(args.db_path)
        
        # Get query embedding
        query_embed = None
        if args.text:
            query_embed = encoder.encode_text(args.text)
            print(f"Searching for text: '{args.text}'")
        elif args.obj_file:
            query_embed = encoder.encode_obj_file(args.obj_file)
            print(f"Searching for 3D model: {args.obj_file}")
        elif args.pc_file:
            query_embed = encoder.encode_pointcloud_file(args.pc_file)
            print(f"Searching for point cloud: {args.pc_file}")
        else:
            print("Error: Need --text, --obj_file, or --pc_file for search")
            return
        
        # Search database
        results = db.search(query_embed, args.top_k)
        print(f"\nTop {len(results)} similar models:")
        for i, (file_path, similarity, metadata) in enumerate(results):
            print(f"{i+1}. {file_path} (similarity: {similarity:.4f})")
            
    elif args.compare:
        if not args.text or not (args.obj_file or args.pc_file):
            print("Error: --compare requires both --text and (--obj_file or --pc_file)")
            return
        
        # Encode text
        text_embed = encoder.encode_text(args.text)
        
        # Encode 3D model
        if args.obj_file:
            model_embed = encoder.encode_obj_file(args.obj_file)
            model_path = args.obj_file
        else:
            model_embed = encoder.encode_pointcloud_file(args.pc_file)
            model_path = args.pc_file
        
        # Compute similarity
        similarity = encoder.compute_similarity(text_embed, model_embed)
        print(f"Similarity between '{args.text}' and '{model_path}': {similarity:.4f}")
        
    elif args.obj_dir:
        # Encode directory
        encode_directory(encoder, args.obj_dir, args.output_dir)
        print(f"Encoded all files in {args.obj_dir}, saved to {args.output_dir}")
        
    elif args.obj_file:
        # Encode single OBJ file
        embedding = encoder.encode_obj_file(args.obj_file)
        
        if args.output_file:
            output_path = args.output_file
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            filename = Path(args.obj_file).stem
            output_path = os.path.join(args.output_dir, f"{filename}_embedding.npy")
        
        np.save(output_path, embedding)
        print(f"Embedding saved to {output_path}")
        print(f"Embedding shape: {embedding.shape}")
        
    elif args.pc_file:
        # Encode single point cloud file
        embedding = encoder.encode_pointcloud_file(args.pc_file)
        
        if args.output_file:
            output_path = args.output_file
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            filename = Path(args.pc_file).stem
            output_path = os.path.join(args.output_dir, f"{filename}_embedding.npy")
        
        np.save(output_path, embedding)
        print(f"Embedding saved to {output_path}")
        print(f"Embedding shape: {embedding.shape}")
        
    elif args.text:
        # Encode text
        embedding = encoder.encode_text(args.text)
        
        if args.output_file:
            output_path = args.output_file
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            # Create safe filename from text
            safe_filename = "".join(c for c in args.text if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_filename = safe_filename.replace(' ', '_')[:50]  # Limit length
            output_path = os.path.join(args.output_dir, f"{safe_filename}_text_embedding.npy")
        
        np.save(output_path, embedding)
        print(f"Text embedding saved to {output_path}")
        print(f"Embedding shape: {embedding.shape}")
        
    else:
        print("Error: Must specify input (--obj_file, --obj_dir, --pc_file, or --text)")
        parser.print_help()


if __name__ == "__main__":
    main()
