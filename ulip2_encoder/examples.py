#!/usr/bin/env python3
"""
Example usage of ULIP2 Encoder

This script demonstrates how to use ULIP2 for encoding 3D models and text descriptions
for similarity-based retrieval.
"""

import os
import sys
import numpy as np

# Add the parent directory to Python path to access ULIP modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from ulip2_encoder import ULIP2Encoder, VectorDatabase


def example_single_obj_encoding():
    """Example: Encode a single OBJ file"""
    print("=== Example 1: Encoding a single OBJ file ===")
    
    # Initialize encoder
    model_path = "ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    encoder = ULIP2Encoder(model_path, device='cuda', num_points=10000)
    
    # Encode an OBJ file
    obj_file = "data/custom_data/data/rectangular.obj"  # Example OBJ file
    if os.path.exists(obj_file):
        embedding = encoder.encode_obj_file(obj_file)
        print(f"Encoded {obj_file}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        # Save embedding
        np.save("rectangular_embedding.npy", embedding)
        print("Saved embedding to rectangular_embedding.npy")
    else:
        print(f"OBJ file not found: {obj_file}")


def example_text_encoding():
    """Example: Encode text descriptions"""
    print("\n=== Example 2: Encoding text descriptions ===")
    
    # Initialize encoder
    model_path = "ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    encoder = ULIP2Encoder(model_path, device='cuda', num_points=10000)
    
    # Encode various text descriptions
    texts = [
        "a rectangular wooden box",
        "a red chair", 
        "a modern table",
        "a lamp with a white shade"
    ]
    
    for text in texts:
        embedding = encoder.encode_text(text)
        print(f"Text: '{text}'")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print()


def example_similarity_comparison():
    """Example: Compare 3D model with text descriptions"""
    print("\n=== Example 3: Similarity comparison ===")
    
    # Initialize encoder
    model_path = "ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    encoder = ULIP2Encoder(model_path, device='cuda', num_points=10000)
    
    # Load a 3D model
    obj_file = "data/custom_data/data/rectangular.obj"
    if not os.path.exists(obj_file):
        print(f"OBJ file not found: {obj_file}")
        return
    
    model_embedding = encoder.encode_obj_file(obj_file)
    
    # Compare with different text descriptions
    test_texts = [
        "a rectangular box",
        "a wooden container", 
        "a square table",
        "a round ball",
        "a chair",
        "a lamp"
    ]
    
    print(f"Comparing {obj_file} with text descriptions:")
    print("-" * 50)
    
    similarities = []
    for text in test_texts:
        text_embedding = encoder.encode_text(text)
        similarity = encoder.compute_similarity(model_embedding, text_embedding)
        similarities.append((text, similarity))
        print(f"'{text}': {similarity:.4f}")
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\nBest match: '{similarities[0][0]}' (similarity: {similarities[0][1]:.4f})")


def example_vector_database():
    """Example: Build and search a vector database"""
    print("\n=== Example 4: Vector database ===")
    
    # Initialize encoder
    model_path = "ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    encoder = ULIP2Encoder(model_path, device='cuda', num_points=10000)
    
    # Build database from custom data directory
    db = VectorDatabase()
    data_dir = "data/custom_data/data"
    
    if os.path.exists(data_dir):
        print(f"Building vector database from {data_dir}")
        
        # Find all OBJ files
        obj_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith('.obj'):
                    obj_files.append(os.path.join(root, file))
        
        print(f"Found {len(obj_files)} OBJ files")
        
        # Encode each file and add to database
        for obj_file in obj_files:
            try:
                embedding = encoder.encode_obj_file(obj_file)
                metadata = {
                    'filename': os.path.basename(obj_file),
                    'full_path': obj_file
                }
                db.add_embedding(embedding, obj_file, metadata)
                print(f"Added {os.path.basename(obj_file)} to database")
            except Exception as e:
                print(f"Error processing {obj_file}: {e}")
        
        # Save database
        db.save("example_vector_db.pkl")
        
        # Search with text query
        if len(db.embeddings) > 0:
            print("\nSearching database with text queries:")
            test_queries = [
                "a box",
                "furniture", 
                "rectangular object"
            ]
            
            for query in test_queries:
                print(f"\nQuery: '{query}'")
                query_embedding = encoder.encode_text(query)
                results = db.search(query_embedding, top_k=3)
                
                for i, (file_path, similarity, metadata) in enumerate(results):
                    filename = metadata.get('filename', os.path.basename(file_path))
                    print(f"  {i+1}. {filename}: {similarity:.4f}")
    else:
        print(f"Data directory not found: {data_dir}")


def example_batch_processing():
    """Example: Batch process all OBJ files in a directory"""
    print("\n=== Example 5: Batch processing ===")
    
    # Initialize encoder
    model_path = "ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    encoder = ULIP2Encoder(model_path, device='cuda', num_points=10000)
    
    # Process all OBJ files in custom data directory
    input_dir = "data/custom_data/data"
    output_dir = "embeddings"
    
    if os.path.exists(input_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all OBJ files
        obj_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.obj'):
                    obj_files.append(os.path.join(root, file))
        
        print(f"Processing {len(obj_files)} OBJ files")
        
        for obj_file in obj_files:
            try:
                # Encode the file
                embedding = encoder.encode_obj_file(obj_file)
                
                # Save embedding
                filename = os.path.basename(obj_file)
                output_file = os.path.join(output_dir, f"{filename}_embedding.npy")
                np.save(output_file, embedding)
                
                print(f"Processed {filename} -> {output_file}")
                
            except Exception as e:
                print(f"Error processing {obj_file}: {e}")
    else:
        print(f"Input directory not found: {input_dir}")


def main():
    """Run all examples"""
    print("ULIP2 Encoder Examples")
    print("=" * 50)
    
    # Check if model exists
    model_path = "ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please download the ULIP2 model first:")
        print("wget https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
        return
    
    try:
        # Run examples (comment out any you don't want to run)
        example_single_obj_encoding()
        example_text_encoding()
        example_similarity_comparison()
        example_vector_database()
        example_batch_processing()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
