Downloaded model: 
`wget https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt` to `models/pointbert`

Downloaded encoder issue zip from: https://github.com/salesforce/ULIP/issues/78
https://github.com/user-attachments/files/19534603/ULIP.zip
and applied changes under `data/` and `encoder_features.py`

Put some mesh OBJ to `data/custom_data/`

Modified encode py to output a numpy save

updated dockerfile to copy EVERYTHING

updated `models/pointbert/misc.py` line 10 to use relative import `from ..pointnet2 import pointnet2_utils`

`pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl`
`pip install Ninja`

downloaded model weights from https://huggingface.co/datasets/SFXX/ulip/tree/main/ULIP-1/initialize_models
`data/initialize_models/`

Other weights (shapetnet-55) are not found (and modelnet contains too many files to wget manually), but maybe we only need these

Now inference takes forever, debug or retry with older model gpu or less obj files (it was just 3...)
