from pathlib import Path

DATASET_ID = "MITLL/LADI-v2-dataset"
HF_CONFIG = "v2a_resized"
HF_REVISION = "script"

LABEL_COLS_V2A = [
    "bridges_any",
    "buildings_any",
    "buildings_affected_or_greater",
    "buildings_minor_or_greater",
    "debris_any",
    "flooding_any",
    "flooding_structures",
    "roads_any",
    "roads_damage",
    "trees_any",
    "trees_damage",
    "water_any",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CACHE_DIR = Path("/kaggle/working/ladi_v2a_cache")
DEFAULT_OUT_DIR = Path("/kaggle/working/ladi_v2_outputs")
DEFAULT_HF_BASE_DIR = Path("/kaggle/working/ladi_hf")

IMG_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]
