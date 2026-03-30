'''
CLI for PENNE inference
'''

import argparse
import torch
from torch.utils.data import DataLoader
from penne import init_spaghetti, pre_processing_phikon, InferenceDataset, Penne
import os

def main():
    parser = argparse.ArgumentParser(description="PENNE Inference CLI")
    parser.add_argument("--input_dir", required=True, help="List of directories containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output gene expression predictions")
    parser.add_argument("--penne_ckpt", default="assets/penne.ckpt", help="Path to the PENNE model checkpoint")
    parser.add_argument("--spaghetti_ckpt", default="assets/spaghetti.ckpt", help="Path to the SPAGHETTI model checkpoint")
    args = parser.parse_args()

    # Initialize the SPAGHETTI model and the PENNE model
    spaghetti_model = init_spaghetti(args.spaghetti_ckpt)
    penne = Penne(spaghetti_model_path=args.spaghetti_ckpt, penne_model_path=args.penne_ckpt)

    # Create the inference dataset and dataloader
    dataset = InferenceDataset(args.input_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    penne.forward(dataloader, args.output_dir)

if __name__ == "__main__":
    main()