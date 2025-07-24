from mylmeval.infer import get_results
from mylmeval.utils import parse_json, open_json, save_json
import glob
import os
import fire
import pandas as pd
import json
import numpy as np

def main(
    model_name_or_path: str = "Qwen/Qwen2.5-VL-32B-Instruct",
    data_name: str = 'example_personality_prompts',
    max_model_len: int = 8192,
    max_num_seqs: int = 10,
    gpu_memory_utilization: float = 0.9,
    only_parsing: bool = False,
):
    data_path = f"save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_{data_name}"
    save_path = os.path.join(data_path, 'vlm_eval', f'{model_name_or_path.replace("/", "_")}.jsonl')
    if only_parsing:
        results = open_json(save_path)
        results = [{**r, 'parsed_result': parse_json(r['result'])} for r in results]
        save_json(results, save_path)
        print(f"Motion accuracy: {max(results, key=lambda x: x['parsed_result']['motion_accuracy'])}")
        print(f"Personality consistency: {max(results, key=lambda x: x['parsed_result']['personality_consistency'])}")
        return
    
    videos = sorted(glob.glob(f"{data_path}/*.mp4"))
    prompts = open('assets/example_personality_prompts.txt').readlines()
    pr_to_vid = len(prompts) // len(videos)
    prompts = ['\n'.join(prompts[i:i+pr_to_vid]) for i in range(0, len(prompts), pr_to_vid)]
    data = []
    if not os.path.exists(os.path.join(data_path, 'vlm_eval')):
        os.makedirs(os.path.join(data_path, 'vlm_eval'))
    for video, prompt in zip(videos, prompts):
        data.append({
            'video_path': video,
            'inputs': [prompt]
        })
    prompt = """You are evaluating motions of a person performing an action based on a given text prompt and persona.

TEXT PROMPT: {0}

Your task is to evaluate the motions of a person in the video on two criteria using a 1-5 scale:

EVALUATION CRITERIA:
1. Motion Accuracy (1-5): How well does the generated motion match the text prompt?
   - 1: Motion completely unrelated to prompt
   - 2: Motion partially matches but has major inconsistencies  
   - 3: Motion generally matches but lacks some details
   - 4: Motion closely matches with minor deviations
   - 5: Motion perfectly matches the prompt

2. Personality Consistency (1-5): How well does the motion reflect the given persona characteristics?
   - 1: No personality traits visible in motion
   - 2: Few personality traits reflected
   - 3: Some personality traits visible but inconsistent
   - 4: Most personality traits well-reflected
   - 5: Personality perfectly embodied in motion

RESPONSE FORMAT:
- Respond with ONLY valid JSON
- Use exact integer values (1, 2, 3, 4, or 5)
- Include the prompt text exactly as provided
- Evaluate each motion separately with indexing (e.g., motion_1, motion_2, ...)
- Do not add any explanations, comments, markdown, or extra text

EXAMPLE RESPONSE:
[{{
    "motion_1": {{
        "motion_accuracy": 4,
        "personality_consistency": 3,
        "prompt": "a person walking confidently"
    }},
    "motion_2": {{
        "motion_accuracy": 2,
        "personality_consistency": 1,
        "prompt": "a person walking confidently"
    }},
    ...
}}]

NOW EVALUATE THE VIDEO AND RESPOND IN THIS EXACT FORMAT:"""
    if os.path.exists(save_path):
        os.remove(save_path)

    _ = get_results(
        model_name_or_path=model_name_or_path,
        data=data,
        prompt=prompt,
        vlm=True,
        # dtype='half',
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        save_path=save_path,
    )
    
    results = open_json(save_path)
    results = [{**r, 'parsed_result': parse_json(r['result'])} for r in results]
    save_json(results, save_path)


def see_score():
    data_path = f"save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_example_personality_prompts"
    
    # Load 3 JSON files with parsed results
    json_files = [
        "save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_example_personality_prompts/vlm_eval/Qwen_Qwen2.5-VL-3B-Instruct.jsonl",
        "save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_example_personality_prompts/vlm_eval/Qwen_Qwen2.5-VL-7B-Instruct.jsonl", 
        "save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10_example_personality_prompts/vlm_eval/Qwen_Qwen2.5-VL-32B-Instruct.jsonl"
    ]
    
    # Load and parse the JSON files
    all_data = []
    for file_path in json_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
                all_data.append(data)
        else:
            print(f"File not found: {file_path}")
            return
    
    data = []
    for i in range(len(all_data[0])):
        for ii in range(3):
            instance = {'idx': i * 3 + ii}
            for d, name in zip(all_data, ['3', '7', '32']):
                try:
                    if isinstance(d[i]['parsed_result'], list):
                        try:
                            model_acc = d[i]['parsed_result'][ii]['motion_accuracy']
                            model_personality = d[i]['parsed_result'][ii]['personality_consistency']
                        except:
                            model_acc = d[i]['parsed_result'][ii][f'motion_{ii+1}']['motion_accuracy']
                            model_personality = d[i]['parsed_result'][ii][f'motion_{ii+1}']['personality_consistency']
                    elif isinstance(d[i]['parsed_result'], dict):
                        model_acc = d[i]['parsed_result'][f'motion_{ii+1}']['motion_accuracy']
                        model_personality = d[i]['parsed_result'][f'motion_{ii+1}']['personality_consistency']
                except:
                    model_acc = -1
                    model_personality = -1
                instance[f'{name}m'] = model_acc
                instance[f'{name}p'] = model_personality
            instance['m_mean'] = np.mean([instance[f'{name}m'] for name in ['3', '7', '32']])
            instance['p_mean'] = np.mean([instance[f'{name}p'] for name in ['3', '7', '32']])
            instance['m_std'] = np.std([instance[f'{name}m'] for name in ['3', '7', '32']])
            instance['p_std'] = np.std([instance[f'{name}p'] for name in ['3', '7', '32']])
            data.append(instance)

    df = pd.DataFrame(data)
    
    # Print as markdown
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    fire.Fire({
        'main': main,
        'see_score': see_score,
    })