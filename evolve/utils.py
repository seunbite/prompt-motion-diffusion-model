import json

def open_json(path: str):
    if path.endswith('.jsonl'):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]
    else:
        with open(path, 'r') as f:
            return json.load(f)
    
def save_json(data: list, path: str):
    if path.endswith('.jsonl'):
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            
            
import re

def extract_num(s: str):
    s = str(s)
    return int(re.search(r'\d+', s).group())