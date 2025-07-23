import argparse
import torch
import numpy as np
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils.utils import get_model, get_dataset
from data.dataset_3d import *

def parse_args():
    parser = argparse.ArgumentParser(description="ULIP-2 Inference Script")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the pretrained checkpoint')
    parser.add_argument('--model', type=str, default='ULIP_PointBERT', help='Model architecture to use')
    parser.add_argument('--output_path', type=str, default='3d_features.npy', help='Path to save the extracted features')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval ulip only')
    parser.add_argument('--evaluate_3d_ulip2', action='store_true', help='eval ulip2 only')
    # EDIT
    parser.add_argument('--validate_dataset_name', default='customdata', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    parser.add_argument('--use_height', action='store_true', help='whether to use height information, by default enabled with PointNeXt.')
    return parser.parse_args()

def load_model(args):
    """Load the ULIP-2 model from a checkpoint"""
    print(f"Loading model {args.model} from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}

    model = getattr(models, args.model)(args=args)
    model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()
    return model

def extract_features(model, args):
    """Extract features from 3D models using the ULIP-2 model."""
    tokenizer = SimpleTokenizer()
    dataset = get_dataset(None, tokenizer, args, 'val')

    # Save each point cloud in the dataset (possibly down-sampled) to an external PLY file
    # The routine save_pc() was added to the data.dataset_3d module
    #for idx in range(len(dataset.dataset.list_of_names)):
    #    save_pc(dataset.dataset.list_of_points[idx], dataset.dataset.list_of_names[idx])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    features_list = []
    with torch.no_grad():
        for pc, name in data_loader:
            pc = pc.cuda()
            features = get_model(model).encode_pc(pc)
            features = features / features.norm(dim=-1, keepdim=True)
            features_list.append(features.cpu().numpy())

    return np.array(features_list)

def main():
    args = parse_args()
    model = load_model(args)
    features = extract_features(model, args)

    # Save the features for later use
    np.save(args.output_path, features)
    print(f"Saved extracted features to {args.output_path}")


if __name__ == '__main__':
    main()