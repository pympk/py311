import json
import os

input_path = r'C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR\GOLDEN_bot_v63u.ipynb'
output_path = r'C:\Users\ping\Files_win10\python\py311\stocks\notebooks_RLVR_v2\GOLDEN_bot_v63u_v2.ipynb'

with open(input_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace notebooks_RLVR with notebooks_RLVR_v2
            line = line.replace('notebooks_RLVR', 'notebooks_RLVR_v2')
            new_source.append(line)
        
        import_block_start = -1
        for i, line in enumerate(new_source):
            if 'from core.analyzer' in line:
                import_block_start = i
                break
        
        if import_block_start != -1:
            import_block_end = -1
            for i in range(import_block_start, len(new_source)):
                if 'from strategy.registry' in new_source[i]:
                    import_block_end = i
                    break
            
            if import_block_end != -1:
                new_imports = [
                    'from core.analyzer import create_walk_forward_analyzer, run_headless_simulation\n',
                    'from core.kernel import FilterPack, EngineInput, MarketObservation, QuantUtils, TickerEngine\n',
                    'from core.config import GLOBAL_SETTINGS, OUTPUT_DIR\n',
                    'from core.environment import ParallelFeatureBuilder, FeatureCubeStitcher, DiscoveryEnv, AlphaLogic\n',
                    'from core.engine import AlphaEngine, AlphaCache\n',
                    'from core.features import generate_features\n',
                    'from core.utils import SystemUtils as SU\n',
                    'from strategy.registry import STRATEGY_REGISTRY\n'
                ]
                # Filter out lines that might be part of the old import block but weren't matched
                # (e.g. core.auditor, core.builder, etc. which were in between)
                # But my logic above already identifies start and end.
                # Let's double check if there are any other 'from core.' in the new_source between start and end.
                # Yes, there are.
                
                updated_source = new_source[:import_block_start] + new_imports + new_source[import_block_end+1:]
                cell['source'] = updated_source
            else:
                cell['source'] = new_source
        else:
            cell['source'] = new_source

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
