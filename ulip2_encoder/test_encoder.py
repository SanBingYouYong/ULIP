#!/usr/bin/env python3
"""
Test script for ULIP2 Encoder

This script tests the encoder with the existing test data in the repository.
Run this to verify everything is working correctly.
"""

import os
import sys
import numpy as np

# Add the parent directory to Python path to access ULIP modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import open3d as o3d
        print(f"✓ Open3D {o3d.__version__}")
    except ImportError:
        print("✗ Open3D not found")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    try:
        from utils.tokenizer import SimpleTokenizer
        print("✓ ULIP tokenizer")
    except ImportError:
        print("✗ ULIP tokenizer not found")
        return False
    
    try:
        import models.ULIP_models as models
        print("✓ ULIP models")
    except ImportError:
        print("✗ ULIP models not found")
        return False
    
    return True


def test_model_loading():
    """Test if the ULIP2 model can be loaded"""
    print("\nTesting model loading...")
    
    model_path = "ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        print("Please download the model using:")
        print("wget https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt -O ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
        return False
    
    try:
        from ulip2_encoder import ULIP2Encoder
        encoder = ULIP2Encoder(model_path, device='cuda' if torch.cuda.is_available() else 'cpu', num_points=1000)
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def test_obj_processing():
    """Test OBJ file processing"""
    print("\nTesting OBJ file processing...")
    
    # Check for test OBJ files
    test_files = [
        "data/also_rectangular.obj",
        "data/disconnected_lamp.obj",
        "data/custom_data/data/rectangular.obj"
    ]
    
    found_obj = None
    for obj_file in test_files:
        if os.path.exists(obj_file):
            found_obj = obj_file
            break
    
    if not found_obj:
        print("✗ No test OBJ files found")
        print("Available test files should be at:")
        for f in test_files:
            print(f"  {f}")
        return False
    
    try:
        from ulip2_encoder import OBJProcessor
        processor = OBJProcessor(num_points=1000)
        points = processor.load_obj_to_pointcloud(found_obj)
        print(f"✓ Successfully processed {found_obj}")
        print(f"  Point cloud shape: {points.shape}")
        print(f"  Point range: [{points[:, :3].min():.3f}, {points[:, :3].max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Error processing OBJ file: {e}")
        return False


def test_encoding():
    """Test encoding functionality"""
    print("\nTesting encoding...")
    
    model_path = "ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    
    if not os.path.exists(model_path):
        print("✗ Model not available for encoding test")
        return False
    
    try:
        from ulip2_encoder import ULIP2Encoder
        
        # Use CPU and fewer points for testing
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = ULIP2Encoder(model_path, device=device, num_points=1000)
        
        # Test text encoding
        text_embedding = encoder.encode_text("a wooden chair")
        print(f"✓ Text encoding successful")
        print(f"  Text embedding shape: {text_embedding.shape}")
        print(f"  Text embedding norm: {np.linalg.norm(text_embedding):.4f}")
        
        # Test point cloud encoding with dummy data
        dummy_points = np.random.randn(1000, 3) * 0.5
        pc_embedding = encoder.encode_pointcloud(dummy_points)
        print(f"✓ Point cloud encoding successful")
        print(f"  PC embedding shape: {pc_embedding.shape}")
        print(f"  PC embedding norm: {np.linalg.norm(pc_embedding):.4f}")
        
        # Test similarity computation
        similarity = encoder.compute_similarity(text_embedding, pc_embedding)
        print(f"✓ Similarity computation successful")
        print(f"  Similarity: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during encoding: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_database():
    """Test vector database functionality"""
    print("\nTesting vector database...")
    
    try:
        from ulip2_encoder import VectorDatabase
        
        # Create test database
        db = VectorDatabase()
        
        # Add some dummy embeddings
        for i in range(3):
            embedding = np.random.randn(1, 1280)  # ULIP2 embedding dimension
            file_path = f"test_model_{i}.obj"
            metadata = {"index": i, "type": "test"}
            db.add_embedding(embedding, file_path, metadata)
        
        print(f"✓ Added {len(db.embeddings)} embeddings to database")
        
        # Test search
        query_embedding = np.random.randn(1, 1280)
        results = db.search(query_embedding, top_k=2)
        print(f"✓ Search returned {len(results)} results")
        
        # Test save/load
        test_db_path = "test_db.pkl"
        db.save(test_db_path)
        print("✓ Database saved successfully")
        
        new_db = VectorDatabase()
        new_db.load(test_db_path)
        print("✓ Database loaded successfully")
        print(f"  Loaded {len(new_db.embeddings)} embeddings")
        
        # Clean up
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing vector database: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("ULIP2 Encoder Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("OBJ Processing Test", test_obj_processing),
        ("Encoding Test", test_encoding),
        ("Vector Database Test", test_vector_database)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The ULIP2 encoder is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        
        if passed == 0:
            print("\nTroubleshooting tips:")
            print("1. Make sure you're in the ULIP conda environment")
            print("2. Download the ULIP2 model weights")
            print("3. Ensure all dependencies are installed")


if __name__ == "__main__":
    # Import torch here to check availability
    try:
        import torch
        print(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not available")
    
    main()
