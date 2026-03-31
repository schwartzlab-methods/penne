'''
Configs for PENNE
'''

from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent
ASSET_DIR = _PACKAGE_DIR / "assets"
SPAGHETTI_MODEL_DIR = ASSET_DIR / "spaghetti.ckpt"
PENNE_MODEL_DIR = ASSET_DIR / "penne.ckpt"
GENE_NAMES_FILE = ASSET_DIR / "gene_names.txt"
HIGH_CONFIDENCE_GENES_FILE = ASSET_DIR / "high_confidence_genes.txt"