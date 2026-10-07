import json
import argparse
from pathlib import Path


def extract_notebook(ipynb_path):
    """Parse a single notebook and return its cell content as a list of lines."""
    ipynb = Path(ipynb_path)
    if not ipynb.exists():
        raise FileNotFoundError(f"Notebook not found: {ipynb}")

    with open(ipynb, "r", encoding="utf-8") as f:
        nb = json.load(f)

    lines = []

    # Notebook header
    lines.append(f"\n{'#'*78}")
    lines.append(f"## NOTEBOOK: {ipynb.name}")
    lines.append(f"## PATH: {ipynb.resolve()}")
    lines.append(f"{'#'*78}\n")

    for i, cell in enumerate(nb.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))

        lines.append(f"\n{'─'*78}")
        lines.append(f"### CELL {i}  |  {cell_type.upper()}")
        lines.append(f"{'─'*78}\n")

        if cell_type == "markdown":
            lines.append(source.strip())

        elif cell_type == "code":
            lines.append("```python")
            lines.append(source.rstrip())
            lines.append("```")

            # Extract outputs
            outputs = cell.get("outputs", [])
            if outputs:
                lines.append("\n# --- Output ---")
                for out in outputs:
                    out_type = out.get("output_type", "")

                    if out_type == "stream":
                        stream_name = out.get("name", "stdout")
                        text = "".join(out.get("text", []))
                        lines.append(f"# [{stream_name}]")
                        lines.append(text.rstrip())

                    elif out_type in ("display_data", "execute_result"):
                        data = out.get("data", {})
                        if "text/plain" in data:
                            txt = "".join(data["text/plain"])
                            lines.append(f"# [result]")
                            lines.append(txt)
                        elif "text/html" in data:
                            lines.append(f"# [HTML output — truncated]")
                        elif "image/png" in data:
                            lines.append(f"# [PNG image output]")
                        else:
                            lines.append(f"# [display: {list(data.keys())}]")

                    elif out_type == "error":
                        ename = out.get("ename", "Error")
                        evalue = out.get("evalue", "")
                        lines.append(f"# [ERROR] {ename}: {evalue}")
                        tb = "\n".join(out.get("traceback", []))
                        if tb:
                            lines.append(f"# Traceback:\n{tb}")

        else:
            lines.append(f"# (Unsupported cell type: {cell_type})")
            lines.append(source.strip())

    # Notebook footer
    lines.append(f"\n{'#'*78}")
    lines.append(f"## END OF: {ipynb.name}")
    lines.append(f"{'#'*78}\n")

    return lines


def nb2txt(ipynb_paths, out_path=None, merge=False):
    """
    Convert one or more Jupyter notebooks to text.

    Parameters
    ----------
    ipynb_paths : str | Path | list
        Single notebook path, or list of paths.
    out_path : str | Path | None
        Output file path. If None, uses notebook name with .txt extension.
        For merge=True, defaults to 'merged_notebooks.txt'.
    merge : bool
        If True, combine all notebooks into a single file.
        If False, create one .txt per notebook.

    Returns
    -------
    list[str]
        Paths to the created text files.
    """
    if isinstance(ipynb_paths, (str, Path)):
        ipynb_paths = [ipynb_paths]

    ipynb_paths = [Path(p) for p in ipynb_paths]

    created = []

    if merge:
        # Single merged output
        if out_path is None:
            out_path = Path("merged_notebooks.txt")
        else:
            out_path = Path(out_path)

        all_lines = []
        all_lines.append(f"{'='*78}")
        all_lines.append(f"# MERGED NOTEBOOKS CONVERSION")
        all_lines.append(f"# Total notebooks: {len(ipynb_paths)}")
        all_lines.append(f"# {'='*78}")

        for p in ipynb_paths:
            if not p.exists():
                print(f"[SKIP] Not found: {p}")
                continue
            if p.suffix != ".ipynb":
                print(f"[SKIP] Not a notebook: {p}")
                continue

            try:
                cell_lines = extract_notebook(p)
                all_lines.extend(cell_lines)
            except Exception as e:
                print(f"[ERROR] Failed to parse {p}: {e}")
                continue

        all_lines.append(f"\n{'='*78}")
        all_lines.append(f"# END OF MERGED FILE")
        all_lines.append(f"{'='*78}\n")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_lines))

        print(f"[OK] Merged {len(ipynb_paths)} notebooks → {out_path}")
        created.append(str(out_path))

    else:
        # One file per notebook
        for p in ipynb_paths:
            if not p.exists():
                print(f"[SKIP] Not found: {p}")
                continue
            if p.suffix != ".ipynb":
                print(f"[SKIP] Not a notebook: {p}")
                continue

            if out_path is None:
                out = p.with_suffix(".txt")
            else:
                out = Path(out_path)

            try:
                lines = extract_notebook(p)
                with open(out, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print(f"[OK] Converted: {p} → {out}")
                created.append(str(out))
            except Exception as e:
                print(f"[ERROR] Failed to convert {p}: {e}")

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to clean text files"
    )
    parser.add_argument("notebooks", nargs="+", help="One or more .ipynb files")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. For single file: output name. For merge: merged file name.",
    )
    parser.add_argument(
        "-m",
        "--merge",
        action="store_true",
        help="Merge all notebooks into a single text file",
    )
    args = parser.parse_args()

    nb2txt(args.notebooks, out_path=args.output, merge=args.merge)


if __name__ == "__main__":
    main()

# from nb2txt import nb2txt

# # Merge two notebooks into one file
# nb2txt(
#     ["RLVR_Part1_AlphaCache.ipynb", "RLVR_Part2_Training.ipynb"],
#     out_path="combined_RLVR.txt",
#     merge=True,
# )

# # Or merge ALL notebooks in a directory
# from pathlib import Path

# notebooks = sorted(Path(".").glob("*.ipynb"))
# nb2txt(notebooks, out_path="all_notebooks.txt", merge=True)
