import pandas as pd
from dataclasses import is_dataclass, fields
from typing import Any, List, Dict, Optional


def peek(idx: int, reg: List[Dict[str, Any]]) -> Any:
    """
    Displays metadata and RETURNS the object for further use.
    SAFEGUARD: Checks if reg is actually a list from the visualizer.
    """
    if not isinstance(reg, list):
        print(f"❌ Error: Pass the result map (list), not the analyzer object.")
        return None

    if idx < 0 or idx >= len(reg):
        print(f"❌ Index {idx} out of range (0 to {len(reg)-1}).")
        return None

    entry = reg[idx]

    print(f" {'='*60}")
    print(f" 📍 INDEX: [{idx}]")
    print(f" 🏷️  NAME:  {entry.get('name', 'N/A')}")
    print(f" 📂 PATH:  {entry.get('path', 'N/A')}")
    print(f" {'='*60}\n")

    try:
        from IPython.display import display

        display(entry["obj"])
    except ImportError:
        print(entry["obj"])

    return entry["obj"]


def visualize_analyzer_structure(analyzer) -> List[Dict]:
    """
    High-level entry point for the Analyzer.
    Maps the internal data structure of the last simulation run.
    """
    # Check if last_run exists (WalkForwardAnalyzer specific logic)
    last_run = getattr(analyzer, "last_run", None)

    if not last_run:
        print("❌ Audit Aborted: No simulation data found in analyzer.last_run.")
        return []

    return visualize_audit_structure(last_run)


def visualize_audit_structure(obj) -> List[Dict]:
    """
    CORE ENGINE: Generates the Map and returns a Registry.
    """
    id_memory = {}
    registry = []
    output = [
        "====================================================================",
        "🔍 HIGH-TRANSPARENCY AUDIT MAP",
        "====================================================================",
    ]

    def get_icon(val):
        if isinstance(val, pd.DataFrame):
            return "🧮"
        if isinstance(val, pd.Series):
            return "📈"
        if isinstance(val, (list, tuple, dict)):
            return "📂"
        if isinstance(val, pd.Timestamp):
            return "📅"
        if is_dataclass(val):
            return "📦"
        return "🔢" if isinstance(val, (int, float)) else "📄"

    def process(item, name, level=0, path=""):
        indent = "  " * level
        item_id = id(item)
        current_path = f"{path} -> {name}" if path else name

        is_primitive = isinstance(item, (int, float, str, bool, type(None)))

        # Avoid infinite recursion and handle shared references
        if not is_primitive and item_id in id_memory:
            output.append(
                f"{indent}          ╰── {name} --> [See ID {id_memory[item_id]}]"
            )
            return

        curr_idx = len(registry)
        registry.append({"name": name, "path": current_path, "obj": item})

        if not is_primitive:
            id_memory[item_id] = curr_idx

        # Generate Metadata String
        meta = f"{type(item).__name__}"
        if hasattr(item, "shape"):
            meta = f"shape={item.shape}"
        elif isinstance(item, (list, dict)):
            meta = f"len={len(item)}"

        output.append(f"[{curr_idx:>3}] {indent}{get_icon(item)} {name} ({meta})")

        # Recursion Logic
        if isinstance(item, dict):
            for k, v in item.items():
                process(v, k, level + 1, current_path)
        elif isinstance(item, (list, tuple)):
            for i, v in enumerate(item):
                process(v, f"index_{i}", level + 1, current_path)
        elif is_dataclass(item):
            for f in fields(item):
                process(getattr(item, f.name), f.name, level + 1, current_path)

    process(obj, "audit_pack")
    print("\n".join(output))

    return registry


#
