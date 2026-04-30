'''
CLI for PENNE inference
'''

import argparse
import torch
from torch.utils.data import DataLoader
from penne import init_spaghetti, pre_processing_phikon, InferenceDataset, Penne
from penne._configs import ASSET_DIR, SPAGHETTI_MODEL_DIR, PENNE_MODEL_DIR, GENE_NAMES_FILE
from penne._utils import download_asset_from_github
import os

def main():
    parser = argparse.ArgumentParser(description="PENNE Inference CLI")
    parser.add_argument("--input_dir", required=True, help="List of directories containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output gene expression predictions")
    parser.add_argument("--gene_names", default=None, help="Path to the gene names file (optional)")
    parser.add_argument("--penne_ckpt", default=None, help="Path to the PENNE model checkpoint (optional)")
    parser.add_argument("--spaghetti_ckpt", default=None, help="Path to the SPAGHETTI model checkpoint (optional)")
    args = parser.parse_args()

    # download models if not provided
    if args.penne_ckpt is None:
        if not PENNE_MODEL_DIR.exists():
            print(f"PENNE model not found at {PENNE_MODEL_DIR}. Downloading from GitHub...")
            download_asset_from_github("penne.ckpt", ASSET_DIR)
        penne_ckpt = PENNE_MODEL_DIR
    else:
        penne_ckpt = args.penne_ckpt
    if args.spaghetti_ckpt is None:
        if not SPAGHETTI_MODEL_DIR.exists():
            print(f"SPAGHETTI model not found at {SPAGHETTI_MODEL_DIR}. Downloading from GitHub...")
            download_asset_from_github("spaghetti.ckpt", ASSET_DIR)
        spaghetti_ckpt = SPAGHETTI_MODEL_DIR
    else:
        spaghetti_ckpt = args.spaghetti_ckpt
    if args.gene_names is None:
        if not GENE_NAMES_FILE.exists():
            print(f"Gene names file not found at {GENE_NAMES_FILE}. Downloading from GitHub...")
            download_asset_from_github("gene_names.txt", ASSET_DIR)
        gene_names = GENE_NAMES_FILE
    else:
        gene_names = args.gene_names

    # Initialize the SPAGHETTI model and the PENNE model
    penne = Penne(spaghetti_model_path=spaghetti_ckpt, penne_model_path=penne_ckpt, gene_names=gene_names)

    # Create the inference dataset and dataloader
    dataset = InferenceDataset(args.input_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    penne.forward(dataloader, args.output_dir)

if __name__ == "__main__":
    main()