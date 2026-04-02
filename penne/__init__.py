"""
PENNE - Phase-to-Expression Neural Network Estimator

A PyTorch implementation for inferring gene expression from phase-contrast microscopy images.
"""

from penne._utils import init_spaghetti, pre_processing_phikon
from penne.model import Penne, TrainPenne
from penne.dataset import InferenceDataset, TrainingDataset
from pathlib import Path
from penne import _modules
from penne._configs import *

__version__ = "1.0.0"

__all__ = [
    "init_spaghetti",
    "pre_processing_phikon",
    "Penne",
    "TrainPenne",
    "InferenceDataset",
    "TrainingDataset",
]

