from pathlib import Path
import json
import black

src_path = Path("stocks/notebooks_RLVR/Untitled7.ipynb")
out_path = Path("stocks/notebooks_RLVR/Untitled7.formatted.ipynb")
nb = json.loads(src_path.read_text(encoding="utf-8"))
changed = 0
for cell in nb.get("cells", []):
    if (
        cell.get("cell_type") == "code"
        and cell.get("metadata", {}).get("language", "python") == "python"
    ):
        src = "".join(cell.get("source", []))
        try:
            formatted = black.format_str(src, mode=black.Mode())
        except Exception:
            formatted = src
        cell["source"] = formatted.splitlines(keepends=True)
        changed += 1

out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"WROTE:{out_path} CHANGED:{changed}")
