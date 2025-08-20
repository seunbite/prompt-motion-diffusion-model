from utils import open_json
from termcolor import colored
import random

def main(
    data_path: str = 'save/motion_verb_score.jsonl',
    top_k: int = 20,
    do_random: bool = True,
):
    data = open_json(data_path)
    keys = list(data[0].keys())
    if do_random:
        data = random.sample(data, top_k)
    else:
        data = data[:top_k]
    for i, item in enumerate(data):
        for key in keys:
            print(colored(key, 'green'), item[key])
        print('---------')
    

if __name__ == '__main__':
    import fire
    fire.Fire(main)