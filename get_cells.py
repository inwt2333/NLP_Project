import json
with open('main.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    print("--- ID:", cell.get('id'))
    print(''.join(cell['source']))
