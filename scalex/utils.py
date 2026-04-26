"""Package-wide constants and small shared utilities."""

# Preprocessing defaults
DEFAULT_TARGET_SUM: int = 10_000
DEFAULT_N_TOP_FEATURES: int = 2_000
DEFAULT_MIN_FEATURES_RNA: int = 600
DEFAULT_MIN_FEATURES_ATAC: int = 100
DEFAULT_MIN_CELLS: int = 3
DEFAULT_MIN_CELL_PER_BATCH: int = 10
DEFAULT_CHUNK_SIZE: int = 20_000

# Model defaults
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LR: float = 2e-4
DEFAULT_MAX_ITERATION: int = 30_000
DEFAULT_LATENT_DIM: int = 10

# Clustering defaults
DEFAULT_RESOLUTION: float = 0.3
DEFAULT_N_NEIGHBORS: int = 15

# Data paths
import os
DATA_PATH: str = os.path.expanduser("~") + "/.scalex/"
GENOME_PATH: str = os.path.expanduser("~") + "/.cache/genome/"

# Profile types
PROFILE_RNA: str = "RNA"
PROFILE_ATAC: str = "ATAC"
PROFILE_ADT: str = "ADT"
