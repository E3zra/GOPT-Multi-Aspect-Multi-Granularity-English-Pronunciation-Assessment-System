import sys, json, os

nb_path = '/workspace/gopt/colab/GOPT_GPU.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find cells with model definitions
for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if 'class GOPT(' in src or 'class GOPTNoPhn(' in src or 'class BaselineLSTM(' in src:
            print(src)
            print('# ===CELL END===')