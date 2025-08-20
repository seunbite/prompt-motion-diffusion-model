import pandas as pd
import os
import json
from mylmeval.infer import get_results
from utils import open_json, extract_num
import random

def clean(
    src = '../data/ATOMIC10X.jsonl',
    dst = '../data/ATOMIC10X.cleaned.jsonl',
    log = '../data/ATOMIC10X.badlines.txt'
):

    bad = 0
    with open(src, 'r', encoding='utf-8', errors='replace') as fin, \
        open(dst, 'w', encoding='utf-8') as fout, \
        open(log, 'w', encoding='utf-8') as flog:
        for i, line in enumerate(fin, start=1):
            try:
                json.loads(line)
                fout.write(line)
            except json.JSONDecodeError as e:
                bad += 1
                flog.write(f"line {i}: {e}\n")

    print(f"Cleaned file written to: {dst} (skipped {bad} bad lines)")



def main(
    data_path = '../data/ATOMIC10X.cleaned.jsonl',
    model_name_or_path = 'Qwen/Qwen2.5-32B-Instruct',
    only_parsing: bool = False,
    chunk_size = 10,
    max_model_len: int = 2048,
    max_num_seqs: int = 50,
    batch_size: int = 50,
    gpu_memory_utilization: float = 0.9,
    th_score1: int = 3,
    th_score2: int = 0,
    prompt_version: int = 2,
):
    if only_parsing:
        from collections import Counter
        data = open_json('save/motion_verb_score.jsonl')
        result_list = []
        for item in data:
            motions = [r.split(".")[-1].strip() for r in item['prompt'].split("Motions:")[1].split("\n") if len(r) > 2]
            results = [r.split(".")[-1].strip() for r in item['result'].split("\n") if len(r) > 0]
            for motion, result in zip(motions, results):
                try:
                    score1, score2 = result.split(" / ")
                except:
                    score1 = 0
                    score2 = 5
                if len(motion) > 3 and 'PersonY' not in motion:
                    result_list.append({
                        'motion': motion,
                        'score1' : extract_num(score1),
                        'score2' : extract_num(score2),
                    })
        counter = Counter([r['score1'] for r in result_list])
        for i in range(6):
            print(i, counter[i])
        counter = Counter([r['score2'] for r in result_list])
        for i in range(6):
            print(i, counter[i])
        
        for th_score1 in range(6):
            for th_score2 in range(6):
                filtered_result_list = [r for r in result_list if r['score1'] == th_score1 and r['score2'] == th_score2]
                print(f"Visibility: {th_score1}, Object-interaction: {th_score2}, Count: {len(filtered_result_list)}")
                sampled_result_list = random.sample(filtered_result_list, 2)
                print(', '.join([r['motion'] for r in sampled_result_list]))
            
        with open('save/motion_verb_score.cleaned.jsonl', 'w') as f:
            for r in filtered_result_list:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
        return
        
    df = pd.read_json(data_path, lines=True)
    unique_verbs = df['head'].unique()
    with open('tmp.txt', 'w') as f:
        f.write('\n'.join(unique_verbs))
    print(f"Found {len(df)} lines and {len(unique_verbs)} unique verb")

    prompt_v1 = """For this {} motion verbs, please rate the score of the verbs from 0 to 5.
Without any other words, just return the score. (e.g., "7 / 8")

1. Visible: Is the verb is highly related with a visible motion?
    - 0: No (e.g., "PersonX plans to return the money", "PersonX takes a risk to")
    - 5: Yes (e.g., "PersonX walks to the door", "PersonX repairs a tire")
2. Object-interaction: Is the verb is highly related with an object-interaction motion?
    - 0: No (e.g., "PersonX walks", "PersonX kicks")
    -5: Yes (e.g., "PersonX repairs a tire", "PersonX sets an alarm")
    
Motions:
{}
"""
    prompt_v2 = """For this {} behavior, please paraphrase the behavior into a distinct, concrete, and observable motion.
- You can combine multiple motions to describe the behavior.
- Please use the common motion verbs to describe the behavior, and make it observable.

Motions:
{}
"""

    input_data = []
    for i in range(0, len(unique_verbs), chunk_size):
        chunk = unique_verbs[i:i+chunk_size]
        input_data.append({
            'inputs' : [chunk_size, '\n'.join([f'{i+1}. {verb}' for i, verb in enumerate(chunk)])],
        })
        
    get_results(
        model_name_or_path=model_name_or_path,
        data=input_data,
        prompt=prompt_v1 if prompt_version == 1 else prompt_v2,
        batch_size=batch_size,
        save_path=f'save/motion_verb_score.v{prompt_version}.jsonl',
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
    )


if __name__ == '__main__':
    import fire
    fire.Fire(main)