"""
PENNE - Phase-to-Expression Neural Network Estimator

A PyTorch implementation for inferring gene expression from phase-contrast microscopy images.
"""

from penne._utils import init_spaghetti, pre_processing_phikon
from penne.model import Penne, TrainPenne
from penne.dataset import InferenceDataset, TrainingDataset
from pathlib import Path
from penne import _modules

__all__ = [
    "init_spaghetti",
    "pre_processing_phikon",
    "Penne",
    "TrainPenne",
    "InferenceDataset",
    "TrainingDataset",
]

_PACKAGE_DIR = Path(__file__).parent
asset_path = _PACKAGE_DIR / "assets"
spaghetti_model_path = asset_path / "spaghetti.ckpt"
penne_model_path = asset_path / "penne.ckpt"
gene_names_path = asset_path / "gene_names.txt"
high_confidence_genes_path = asset_path / "high_confidence_genes.txt"