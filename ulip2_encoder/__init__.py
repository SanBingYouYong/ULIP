"""
ULIP2 Encoder Package

This package provides tools for encoding 3D models and text using ULIP2
for similarity-based retrieval applications.

Main components:
- ULIP2Encoder: Main encoder class for 3D models and text
- OBJProcessor: Handles OBJ file loading and point cloud conversion  
- VectorDatabase: Simple vector database for embeddings storage and search

Usage:
    from ulip2_encoder import ULIP2Encoder, VectorDatabase
    
    encoder = ULIP2Encoder(model_path)
    embedding = encoder.encode_obj_file("model.obj")
"""

from .ulip2_encoder import ULIP2Encoder, OBJProcessor, VectorDatabase

__version__ = "1.0.0"
__author__ = "GitHub Copilot"

__all__ = ["ULIP2Encoder", "OBJProcessor", "VectorDatabase"]
