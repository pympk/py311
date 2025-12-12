import json
import datetime

nb_path = r'c:\Users\ping\Files_win10\python\py311\stocks\notebooks_antigravity\bot_v27.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

target_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def plot_walk_forward_analyzer" in source:
            target_cell = cell
            break

if target_cell:
    new_source = []
    lines = target_cell['source']
    
    for line in lines:
        # Fix 1: Force layout width/height and autosize
        if "fig.update_layout(" in line:
            # We replace the entire line to be sure
            new_line = "    fig.update_layout(title='Walk-Forward Performance Analysis', height=600, width=1200, template=\"plotly_white\", hovermode='x unified', autosize=True, margin=dict(l=20, r=20, t=40, b=20))\n"
            new_source.append(new_line)
        
        # Fix 2: Ensure vertical line uses pydatetime
        elif "fig.layout.shapes =" in line:
            # We need to convert res.calc_end_date to pydatetime
            new_line = "                fig.layout.shapes = [dict(type=\"line\", x0=res.calc_end_date.to_pydatetime(), y0=0, x1=res.calc_end_date.to_pydatetime(), y1=1, xref='x', yref='paper', line=dict(color=\"grey\", width=2, dash=\"dash\"))]\n"
            new_source.append(new_line)

        # Fix 3: Add a print to confirm version
        elif "def update_plot(b):" in line:
            new_source.append(line)
            new_source.append("        print('--- Plot Updated (v2) ---')\n")
            
        else:
            new_source.append(line)
            
    target_cell['source'] = new_source
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully (v2).")
else:
    print("Target cell not found.")
