from pathlib import Path

parent_name = "notebooks_RLVR_v2"


def find_notebooks_root(parent_name):
    """Find notebooks_RLVR root from any location."""
    current = Path(__file__).resolve()

    for parent in [current] + list(current.parents):
        if parent.name == parent_name:
            return parent
    raise RuntimeError(f"Could not find {parent_name} directory")


NOTEBOOKS_RLVR_ROOT = find_notebooks_root(parent_name)
OUTPUT_DIR = NOTEBOOKS_RLVR_ROOT / "output"

# Create dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"NOTEBOOKS_RLVR_ROOT: {NOTEBOOKS_RLVR_ROOT}\n")
print(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
