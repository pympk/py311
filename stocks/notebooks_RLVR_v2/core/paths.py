import sys
from pathlib import Path

parent_name = "notebooks_RLVR_v2"


def find_notebooks_root(parent_name):
    """Find notebooks_RLVR root from any location."""
    current = Path(__file__).resolve()

    for parent in [current] + list(current.parents):
        if parent.name == parent_name:
            return parent
    raise RuntimeError(f"Could not find {parent_name} directory")


# Base Directory Resolution
NOTEBOOKS_RLVR_ROOT = find_notebooks_root(parent_name)
PROJECT_ROOT = NOTEBOOKS_RLVR_ROOT.parent

# 1. Global Data Directories (stocks/data)
GLOBAL_DATA_DIR = PROJECT_ROOT / "data"
GLOBAL_PROCESSED_DIR = GLOBAL_DATA_DIR / "processed"

# 2. Local Data Directories (stocks/notebooks_RLVR_v2/data)
LOCAL_DATA_DIR = NOTEBOOKS_RLVR_ROOT / "data"
OUTPUT_DIR = NOTEBOOKS_RLVR_ROOT / "output"

# Ensure necessary directories exist
GLOBAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Environment detection
RUNNING_IN_COLAB = "google.colab" in sys.modules

print(f"NOTEBOOKS_RLVR_ROOT: {NOTEBOOKS_RLVR_ROOT}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"GLOBAL_DATA_DIR: {GLOBAL_DATA_DIR}")
print(f"GLOBAL_PROCESSED_DIR: {GLOBAL_PROCESSED_DIR}")
print(f"LOCAL_DATA_DIR: {LOCAL_DATA_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
