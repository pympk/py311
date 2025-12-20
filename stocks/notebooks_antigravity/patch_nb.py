import json

nb_path = (
    r"c:\Users\ping\Files_win10\python\py311\stocks\notebooks_antigravity\bot_v27.ipynb"
)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The target code is in the cell that defines `plot_walk_forward_analyzer`
# We need to find the cell that contains "def plot_walk_forward_analyzer"

target_cell = None
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def plot_walk_forward_analyzer" in source:
            target_cell = cell
            break

if target_cell:
    new_source = []
    lines = target_cell["source"]

    # We will iterate and replace the specific lines
    for line in lines:
        # Fix 1: Update layout
        if "fig.update_layout(" in line and "hovermode='x unified'" in line:
            new_line = line.replace(
                "hovermode='x unified'",
                "hovermode='x unified', autosize=True, margin=dict(l=20, r=20, t=40, b=20)",
            )
            new_source.append(new_line)

        # Fix 2: Convert indices to pydatetime
        elif "fig.data[i].update(x=res.normalized_plot_data.index" in line:
            # We need to insert the conversion line before this loop if possible, or just do it inline
            # Easier to do it inline for the replacement logic
            new_line = line.replace(
                "x=res.normalized_plot_data.index",
                "x=res.normalized_plot_data.index.to_pydatetime()",
            )
            new_source.append(new_line)

        elif "fig.data[50].update(x=res.benchmark_series.index" in line:
            new_line = line.replace(
                "x=res.benchmark_series.index",
                "x=res.benchmark_series.index.to_pydatetime()",
            )
            new_source.append(new_line)

        elif "fig.data[51].update(x=res.portfolio_series.index" in line:
            new_line = line.replace(
                "x=res.portfolio_series.index",
                "x=res.portfolio_series.index.to_pydatetime()",
            )
            new_source.append(new_line)

        else:
            new_source.append(line)

    target_cell["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
