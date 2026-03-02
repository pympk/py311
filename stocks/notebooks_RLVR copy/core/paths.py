# from pathlib import Path

# # Base paths
# NOTEBOOKS_RLVR_ROOT = Path(__file__).parent.parent

# # Output directories
# OUTPUT_DIR = NOTEBOOKS_RLVR_ROOT / "output"
# # OUTPUT_DATA = OUTPUT_DIR / "data"
# # OUTPUT_PLOTS = OUTPUT_DIR / "plots"
# # OUTPUT_REPORTS = OUTPUT_DIR / "reports"

# create_dirs = [OUTPUT_DIR]

# # Auto-create all directories
# # for d in [OUTPUT_DIR, OUTPUT_DATA, OUTPUT_PLOTS, OUTPUT_REPORTS]:
# for d in create_dirs:
#     d.mkdir(parents=True, exist_ok=True)

###############################

from pathlib import Path


def find_notebooks_root():
    """Find notebooks_RLVR root from any location."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == "notebooks_RLVR":
            return parent
    raise RuntimeError("Could not find notebooks_RLVR directory")


NOTEBOOKS_RLVR_ROOT = find_notebooks_root()
OUTPUT_DIR = NOTEBOOKS_RLVR_ROOT / "output"

# Create dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"NOTEBOOKS_RLVR_ROOT: {NOTEBOOKS_RLVR_ROOT}\n")
print(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
