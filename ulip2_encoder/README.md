# ULIP2 Encoder for 3D Model Retrieval

This tool provides a complete solution for encoding 3D models (.obj files or point clouds) and text descriptions using ULIP2 for similarity-based retrieval. It's designed for applications where you need to compare 3D models with text descriptions or build vector databases for 3D model search.

## Features

- **3D Model Encoding**: Support for .obj, .ply, .pcd, .npy, and .txt point cloud files
- **Text Encoding**: Encode text descriptions using OpenCLIP (ViT-G/14)
- **Similarity Computation**: Calculate cosine similarity between 3D models and text
- **Vector Database**: Build and search vector databases for large-scale 3D model retrieval
- **Batch Processing**: Process entire directories of 3D models
- **Flexible Point Sampling**: Configurable number of points sampled from 3D models

## Installation

Make sure you have the ULIP2 environment set up as described in the main README. The encoder requires:

- PyTorch
- Open3D
- NumPy
- tqdm (optional, for progress bars)

## Quick Start

### 1. Download the ULIP2 Model

```bash
# Download the pretrained ULIP2 model (run from main ULIP directory)
wget https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt -O ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
```

### 2. Basic Usage Examples (run from main ULIP directory)

#### Encode a Single OBJ File
```bash
python ulip2_encoder/ulip2_encoder.py --obj_file data/custom_data/data/rectangular.obj --output_dir embeddings/
```

#### Encode a Text Description
```bash
python ulip2_encoder/ulip2_encoder.py --text "a red wooden chair" --output_dir embeddings/
```

#### Compare 3D Model with Text
```bash
python ulip2_encoder/ulip2_encoder.py --obj_file data/custom_data/data/rectangular.obj --text "a red chair" --compare
```

#### Encode All Files in a Directory
```bash
python ulip2_encoder/ulip2_encoder.py --obj_dir data/custom_data/data/ --output_dir embeddings/
```

#### Build a Vector Database
```bash
python ulip2_encoder/ulip2_encoder.py --obj_dir data/custom_data/data/ --build_db --db_path vector_db.pkl
```

#### Search in Vector Database
```bash
# Search with text
python ulip2_encoder/ulip2_encoder.py --search_db --db_path vector_db.pkl --text "a chair" --top_k 5

# Search with 3D model
python ulip2_encoder/ulip2_encoder.py --search_db --db_path vector_db.pkl --obj_file data/custom_data/data/rectangular.obj --top_k 5
```

## Command Line Arguments

### Model Configuration
- `--model_path`: Path to ULIP2 pretrained model (default: ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt)
- `--device`: Device to use ('cuda' or 'cpu', default: 'cuda')
- `--num_points`: Number of points to sample from 3D models (default: 10000)

### Input Options
- `--obj_file`: Single OBJ file to encode
- `--obj_dir`: Directory containing 3D files
- `--pc_file`: Single point cloud file (.ply, .pcd, .npy, .txt)
- `--text`: Text description to encode

### Output Options
- `--output_dir`: Directory to save embeddings (default: 'embeddings')
- `--output_file`: Specific output file for single embedding

### Vector Database
- `--build_db`: Build vector database from directory
- `--db_path`: Path to save/load vector database (default: 'vector_database.pkl')
- `--search_db`: Search in existing vector database
- `--top_k`: Number of top results to return (default: 5)

### Other Options
- `--compare`: Compare 3D model with text description

## Python API

### Basic Usage

```python
from ulip2_encoder import ULIP2Encoder, VectorDatabase

# Initialize encoder
encoder = ULIP2Encoder(
    model_path="ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt",
    device='cuda',
    num_points=10000
)

# Encode a 3D model
obj_embedding = encoder.encode_obj_file("chair.obj")

# Encode text
text_embedding = encoder.encode_text("a wooden chair")

# Compute similarity
similarity = encoder.compute_similarity(obj_embedding, text_embedding)
print(f"Similarity: {similarity:.4f}")
```

### Vector Database

```python
# Build database
db = VectorDatabase()

# Add embeddings
for obj_file in obj_files:
    embedding = encoder.encode_obj_file(obj_file)
    db.add_embedding(embedding, obj_file, {"type": "furniture"})

# Save database
db.save("furniture_db.pkl")

# Load and search
db = VectorDatabase()
db.load("furniture_db.pkl")

query_embedding = encoder.encode_text("a chair")
results = db.search(query_embedding, top_k=5)

for file_path, similarity, metadata in results:
    print(f"{file_path}: {similarity:.4f}")
```

## Supported 3D File Formats

### OBJ Files (.obj)
- Automatically converts mesh to point cloud
- Supports colored meshes
- Handles both triangular meshes and vertex-only files

### Point Cloud Files
- **.ply**: PLY format point clouds
- **.pcd**: PCL format point clouds  
- **.npy**: NumPy arrays (N×3 or N×6 for XYZ or XYZRGB)
- **.txt**: Text files with point coordinates

## Point Cloud Processing

The encoder automatically:
1. **Normalizes** point clouds to unit sphere
2. **Samples** to specified number of points (default: 10000)
3. **Handles colors** when available (XYZRGB format)
4. **Upsamples** small point clouds by repeating points
5. **Downsamples** large point clouds randomly

## Examples

Run the provided examples to see the encoder in action (from main ULIP directory):

```bash
python ulip2_encoder/examples.py
```

This will demonstrate:
1. Single OBJ file encoding
2. Text description encoding
3. Similarity comparison
4. Vector database creation and search
5. Batch processing

## Use Cases

### 1. 3D Model Search Engine
Build a search engine where users can find 3D models using text descriptions:

```python
# Build database from model collection
encoder = ULIP2Encoder(model_path, device='cuda')
db = build_vector_database(encoder, "model_collection/", "search_db.pkl")

# Search with text
query = "modern office chair"
query_embedding = encoder.encode_text(query)
results = db.search(query_embedding, top_k=10)
```

### 2. Content-Based 3D Model Recommendation
Find similar 3D models based on a query model:

```python
# Encode query model
query_embedding = encoder.encode_obj_file("query_chair.obj")

# Search for similar models
results = db.search(query_embedding, top_k=5)
```

### 3. Automated 3D Model Categorization
Use text descriptions to categorize 3D models:

```python
categories = ["chair", "table", "lamp", "sofa"]
model_embedding = encoder.encode_obj_file("unknown_model.obj")

best_category = None
best_similarity = -1

for category in categories:
    cat_embedding = encoder.encode_text(f"a {category}")
    similarity = encoder.compute_similarity(model_embedding, cat_embedding)
    
    if similarity > best_similarity:
        best_similarity = similarity
        best_category = category

print(f"Model is most likely a {best_category} (similarity: {best_similarity:.4f})")
```

### 4. Cross-Modal Retrieval
Retrieve 3D models using natural language queries:

```python
# Natural language queries
queries = [
    "find me a comfortable office chair",
    "show me modern dining tables", 
    "I need a bedside lamp",
    "looking for outdoor furniture"
]

for query in queries:
    query_embedding = encoder.encode_text(query)
    results = db.search(query_embedding, top_k=3)
    print(f"\nResults for '{query}':")
    for file_path, similarity, metadata in results:
        print(f"  {os.path.basename(file_path)}: {similarity:.4f}")
```

## Performance Tips

1. **GPU Usage**: Use CUDA for faster encoding (default)
2. **Batch Processing**: Process multiple files in one session to amortize model loading time
3. **Point Sampling**: Adjust `num_points` based on your accuracy/speed requirements
4. **Database Size**: For large databases, consider using approximate nearest neighbor search libraries like Faiss

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `num_points` or use CPU
2. **OBJ Loading Errors**: Ensure OBJ files are valid and contain geometry
3. **Model Not Found**: Check model path and download the pretrained weights
4. **Import Errors**: Ensure all dependencies are installed in your environment

### Error Handling

The encoder includes robust error handling:
- Invalid 3D files are replaced with default point clouds
- Missing files are reported with clear error messages
- Database operations include progress tracking

## License

This tool is built on top of ULIP2 and follows the same licensing terms. See the main repository LICENSE file for details.
