# ULIP2 Encoder - Usage Guide

## Running from Main ULIP Directory

All commands should be run from the main ULIP directory (where main.py is located).

### Test Installation
```bash
python ulip2_encoder/test_encoder.py
```

### Run Examples
```bash
python ulip2_encoder/examples.py
```

### Basic Commands

#### Encode a single OBJ file:
```bash
python ulip2_encoder/ulip2_encoder.py --obj_file data/custom_data/data/rectangular.obj
```

#### Encode text:
```bash
python ulip2_encoder/ulip2_encoder.py --text "a wooden chair"
```

#### Compare 3D model with text:
```bash
python ulip2_encoder/ulip2_encoder.py --obj_file data/custom_data/data/rectangular.obj --text "a wooden box" --compare
```

#### Build vector database:
```bash
python ulip2_encoder/ulip2_encoder.py --obj_dir data/custom_data/data/ --build_db --db_path furniture.pkl
```

#### Search database:
```bash
python ulip2_encoder/ulip2_encoder.py --search_db --db_path furniture.pkl --text "a box" --top_k 5
```

### File Structure
```
ULIP/                           # Run commands from here
├── main.py
├── ulip_models/               # Model files go here
├── data/                      # Your 3D data
└── ulip2_encoder/            # Encoder package
    ├── ulip2_encoder.py      # Main encoder
    ├── examples.py           # Examples
    ├── test_encoder.py       # Tests
    └── README.md             # Full documentation
```

### Download Model
```bash
wget https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt -O ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
```

That's it! The encoder is self-contained in the ulip2_encoder/ folder but designed to be run from the parent directory.
