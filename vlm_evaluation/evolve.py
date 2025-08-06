import yaml
import os
from tqdm import tqdm
import gc
import torch
from datetime import datetime
import json
import glob
import multiprocessing as mp
from functools import partial


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

def process_chunk(chunk_data, gpu_id, output_path, now_date, arg, model_path):
    """Worker function to process a chunk of personalities on a specific GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Write chunk to temporary file
    chunk_file = f'tmp_gpu_{gpu_id}.txt'
    with open(chunk_file, 'w') as f:
        for p in chunk_data:
            f.write(f'{p}\n')
    
    print(f"GPU {gpu_id}: Processing {len(chunk_data)} motions...")
    
    # Run the generation command
    cmd = f"micromamba run -n mdm python -m sample.generate --model_path {model_path} --num_samples 3 --num_repetitions 1 --input_text {chunk_file} --output_dir {output_path}/{now_date}_mdm_{arg}/gpu_{gpu_id}"
    os.system(cmd)
    
    print(f"GPU {gpu_id}: Completed processing {len(chunk_data)} motions")
    
    # Clean up temporary file
    os.remove(chunk_file)
    
    # Clean GPU memory
    gc.collect()
    torch.cuda.empty_cache()

personality_motion_set = [
    # Extraversion (10)
    ('extraversion', 'a person walks swinging arms broadly'),
    ('extraversion', 'a person walks with taking big bouncy steps'),
    ('extraversion', 'a person walks chin up'),
    ('extraversion', 'a person walks with clapping hands'),
    ('extraversion', 'a person walks with whistling'),
    ('extraversion', 'a person walks with lifting knees high'),
    ('extraversion', 'a person walks turning to others'),
    ('extraversion', 'a person walks making wide arm circles'),
    ('extraversion', 'a person walks with stamping feet'),
    ('extraversion', 'a person walks with head thrown back laughing'),

    # Agreeableness (10)
    ('agreeableness', 'a person walks giving a gentle wave'),
    ('agreeableness', 'a person walks nodding warmly'),
    ('agreeableness', 'a person walks with relaxing shoulders'),
    ('agreeableness', 'a person walks tilting head with a soft smile'),
    ('agreeableness', 'a person walks with hands clasped softly'),
    ('agreeableness', 'a person walks stepping lightly'),
    ('agreeableness', 'a person walks showing open palms'),
    ('agreeableness', 'a person walks exhaling before each step'),
    ('agreeableness', 'a person walks lowering gaze then looking up'),
    ('agreeableness', 'a person walks patting self gently'),

    # Conscientiousness (10)
    ('conscientiousness', 'a person walks with back straight'),
    ('conscientiousness', 'a person walks placing each foot precisely'),
    ('conscientiousness', 'a person walks with slow measured pace'),
    ('conscientiousness', 'a person walks hands near torso'),
    ('conscientiousness', 'a person walks with no extra sway'),
    ('conscientiousness', 'a person walks pausing between steps'),
    ('conscientiousness', 'a person walks checking posture'),
    ('conscientiousness', 'a person walks with head level'),
    ('conscientiousness', 'a person walks with arms folded'),
    ('conscientiousness', 'a person walks with firm deliberate gestures'),

    # Neuroticism (10)
    ('neuroticism', 'a person walks looking around nervously'),
    ('neuroticism', 'a person walks rubbing hands anxiously'),
    ('neuroticism', 'a person walks tensing then relaxing shoulders'),
    ('neuroticism', 'a person walks shifting weight'),
    ('neuroticism', 'a person walks fidgeting with sleeves'),
    ('neuroticism', 'a person walks biting lip'),
    ('neuroticism', 'a person walks glancing sideways quickly'),
    ('neuroticism', 'a person walks with slight arm shaking'),
    ('neuroticism', 'a person walks with shallow breathing'),
    ('neuroticism', 'a person walks with small head jerks'),

    # Openness (10)
    ('openness', 'a person walks looking up in wonder'),
    ('openness', 'a person walks tracing air shapes with finger'),
    ('openness', 'a person walks swaying to imagined music'),
    ('openness', 'a person walks tilting head to look around'),
    ('openness', 'a person walks reaching arms out to explore'),
    ('openness', 'a person walks stepping then pausing in curiosity'),
    ('openness', 'a person walks with slow spin to observe'),
    ('openness', 'a person walks tracing floor lines with toes'),
    ('openness', 'a person walks with hand on chin thinking'),
    ('openness', 'a person walks arching back to look around'),
]

def load_yaml(path: str):
    personality_yaml = yaml.load(open(os.path.join(path, 'personality.yaml'), 'r'), Loader=yaml.FullLoader)
    motion_yaml = yaml.load(open(os.path.join(path, 'motion.yaml'), 'r'), Loader=yaml.FullLoader)
    personality_with_motions = []
    for category, personalities in personality_yaml.items():
        for motion_name, motions in motion_yaml.items():
            personality_list = personalities.split(", ")
            motion_list = motions.split(", ")
            for personality in personality_list:
                for motion_prompt in motion_list:
                    personality_with_motions.append({
                        'personality': personality,
                        'motion': motion_prompt,
                        'type' : {'motion': motion_name, 'personality': category},
                        'mdm_prompt' : motion_prompt.replace('a person', f'a {personality.lower()} person'),
                        'inputs' : [motion_prompt, personality]
                    })
                    
    open('dataset/prompt/mdm_prompt.txt', 'w').write('\n'.join([f'{p["mdm_prompt"]}' for p in personality_with_motions]))
    return personality_with_motions


def main(
    prompt_path: str = 'dataset/prompt',
    stage: str = 'prompt_momo', # gen_mdm, gen_momo, retarget_motion
    arg: str = 'type1',
    tmp_chunk: int = 20,
    max_model_len: int = 2048,
    max_num_seqs: int = 50,
    gpu_memory_utilization: float = 0.9,
    output_path: str = '/scratch2/iyy1112/motion-persona/save'
):
    now_date = datetime.now().strftime("%Y%m%d")
    personality_with_motions = load_yaml(prompt_path) # [{'personality': 'extrovert', 'motion': 'walk'}, {'personality': 'introvert', 'motion': 'run'}]
    if stage == 'prompt_momo':
        from mylmeval.infer import get_results
        prompt = """You are generating motion prompts that reflect a given personality.
Follow these steps:
1) Imagine a person with the given personality performing the given motion.
2) Modify or extend the motion to clearly reflect how the personality affects their movement or posture.
3) Make each prompt **a concrete and observable physical behavior**, not an abstract trait.

Provide exactly 10 outputs. Each output must:
- Begin with the original [Motion]
- Then add a '+' and a clear, concrete detail that reflects the [Persona]
- Be short sentences describing the combined motion

Format:
[Motion]: <simple base motion>
[Persona]: <personality or style>
[Output]:
1. <motion with personality>
2. ...
3. ...
...

[Example]
[Motion]: a person walks forward  
[Persona]: extrovert  
[Output]:
1. a person walks forward + waves to nearby people  
2. a person walks forward + swinging their arms widely  
3. a person walks forward + bouncing steps  
"""

        _ = get_results(
            model_name_or_path='Qwen/Qwen2.5-32B-Instruct',
            prompt=prompt,
            data=personality_with_motions,
            max_tokens=1000,
            temperature=0.3,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            save_path=os.path.join(prompt_path, 'momo_prompt.jsonl')
        )
    elif stage == 'prompt_augmentation':
        from mylmeval.infer import get_results
        prompt = """You are generating motion prompts with detail (e.g., A person walks with big bouncy steps) to reflect the given personality (e.g., high extraversion).
Please provide 20 different motions prompts for each given personality.

Steps:
1) Imagine a person with the given personality performing the given motion with diversity. (e.g, High extraversion - big bouncy steps, shaking head, shaking hands)
2) Describe the MOTION PROMPTS with ENHANCED DETAIL to clearly reflect the personality type. (Begin with the original [Motion], then add a description of the additional motion detail.)

Constraints:
- Don't specify the inner personality or mind state, just describe the **visible** motion detail.
- For motion generation, the hand motion and face motion are not available.
- Please keep the detailed motion instruction simple andclear with common motions.

Format: 
1. A person walks with big bouncy steps
2. A person walks shaking hands
3. ...

[Motion]: {}
[Persona]: {} {}
[Output]:
1.
"""
        data = []
        persons = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        with open('assets/dataseed/prompt_verb_100.txt', 'r') as f:
            for line in f:
                for personality in persons:
                    for hl in ['high', 'low']:
                        data.append({
                            'inputs': [line, hl.capitalize(), personality.lower()],
                            'hl': hl,
                            'personality': personality,
                            'motion': line,
                        })

        _ = get_results(
            model_name_or_path='Qwen/Qwen2.5-32B-Instruct',
            prompt=prompt,
            data=data,
            batch_size=max_num_seqs,
            max_tokens=1000,
            temperature=0.3,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            save_path=os.path.join(prompt_path, f'{now_date}_{stage}.jsonl')
        )

    elif stage == 'prompt_evaluation':
        from mylmeval.infer import get_results
        prompt = """Please evaluate the generated motion with detailed persona-based motion instruction.
You just say the type number for the motions.
If it is Type 4, please refine the instruction right after the type number. (e.g., 4. <refined instruction>)

Types:
0: good, the motion is walking and also following the detail
1: imperfect, the motion is not walking
2: imperfect, the motion is walking but not following the detail (just walking)
3: the motion is both walking and also following the detail, but somewhat strange
4: imperfect, but if instruction is refined, it can be better
 - It deals with currently unavailable motion detail (e.g., face motion or finger motion).
 - The motion generator misunderstood the instruction, we can paraphrase more.

Format:
[persona-based motion instruction]: {}
[Output]:
"""
        data = []
        data_path = f"{output_path}/mdm/0"
        videos = sorted(glob.glob(f"{data_path}/*.mp4"))
        prompts = [r[1] for r in personality_motion_set]
        for video, prompt in zip(videos, prompts):
            data.append({
                'video_path': video,
                'inputs': [prompt] 
            })
            
        _ = get_results(
            model_name_or_path='Qwen/Qwen2.5-VL-32B-Instruct',
            prompt=prompt,
            data=data,
            max_tokens=1000,
            temperature=0.3,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            save_path=os.path.join(prompt_path, f'{now_date}_{stage}.jsonl')
        )
        
    elif stage == 'gen_mdm':
        os.makedirs(os.path.join(output_path, f'{now_date}_mdm'), exist_ok=True)
        if arg == 'type1':
            personalities = [p[1] for p in personality_motion_set]
        elif arg == 'type2':
            with open(os.path.join('assets/dataseed/prompt_verb_100.txt'), 'r') as f:
                personalities = [p for p in f.readlines()]
            personalities = [p.strip() for p in personalities if len(p) > 3]
        elif arg == 'type3':
            data = open_json(os.path.join(prompt_path, '20250804_prompt_augmentation.jsonl'))
            new_data = [
                [{**item, 'prompt' : r.strip(".").split(". ")[-1]} for r in item['result'].split("\n") if len(r) > 3 and '[' not in r]
                 for item in data]
            new_data = [p for sublist in new_data for p in sublist]
            save_json(new_data, os.path.join(prompt_path, f'prompts.jsonl'))
            personalities = [p['prompt'] for p in new_data]
            print(personalities)
        
        # Split personalities into chunks for each GPU
        num_gpus = torch.cuda.device_count()
        chunk_size = len(personalities) // num_gpus
        chunks = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(personalities)
            chunks.append(personalities[start_idx:end_idx])
        
        print(f"Total motions to process: {len(personalities)}")
        print(f"Using {num_gpus} GPUs with {chunk_size} motions per GPU")
        
        # Create partial function with fixed arguments
        model_path = "./save/humanml_enc_512_50steps/model000750000.pt"
        worker_func = partial(process_chunk, 
                           output_path=output_path, 
                           now_date=now_date, 
                           arg=arg, 
                           model_path=model_path)
        
        # Create processes for each GPU
        processes = []
        for gpu_id in range(num_gpus):
            if chunks[gpu_id]:  # Only create process if chunk is not empty
                p = mp.Process(target=worker_func, args=(chunks[gpu_id], gpu_id))
                processes.append(p)
                p.start()
        
        # Wait for all processes to complete with progress bar
        print(f"Starting generation on {len(processes)} GPUs...")
        with tqdm(total=len(processes), desc="GPU Processes", unit="GPU") as pbar:
            for p in processes:
                p.join()
                pbar.update(1)
        
        print("All GPU processes completed!")


    elif stage == 'retarget_motion':
        motion_files = glob.glob(f"{output_path}/20250805_mdm_type3/*/*/motion.npy")
        if not motion_files:
            print(f"No motion files found in {output_path}/20250805_mdm_type3/")
            return
        
        print(f"Found {len(motion_files)} motion files to retarget")
        
        for motion_file in tqdm(motion_files, desc="Retargeting motions"):
            try:
                dirname = os.path.dirname(motion_file)
                motion_id = os.path.basename(dirname)
                output_file = os.path.join(dirname, f'retarget_{motion_id}.mp4')
                
                # Use subprocess with explicit environment variable passing
                import subprocess
                cmd = ["micromamba", "run", "-n", "mdm", "python", "-m", "sample.retarget_humanoid", 
                       "--motion_file", motion_file, "--output_file", output_file]
                
                # Create environment with MUJOCO_GL set
                env = os.environ.copy()
                env['MUJOCO_GL'] = 'egl'
                
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                
                if result.returncode != 0:
                    print(f"Failed to retarget {motion_id}: {result.stderr}")
                else:
                    print(f"Successfully retargeted {output_file}")
                    
            except Exception as e:
                print(f"Error processing {motion_file}: {e}")
                continue



    elif stage == 'gen_momo':
        # personality_with_motions = open_json(os.path.join(prompt_path, 'momo_prompt.jsonl'))
        # personality_with_motions = [
        #     {**item, 'mdm_prompt': [r.split(": ")[-1] for r in item['result'].split("\n") if '. ' in r]} for item in personality_with_motions
        # ]
        motions = [
            'a person walks forward',
            'a person walks backward',
            'a person kicks',
            'a person jumps',
            'a person jumps',
            ]

        personality_with_motions = [(motion, personality) for motion in motions for personality in personality_motion_set]
        os.makedirs(os.path.join(output_path, 'momo'), exist_ok=True)
        for motion, (personality, personality_motion) in tqdm(personality_with_motions):
            given_motion = f'--text_leader "{motion}" --text_follower "{personality_motion}"'
            cmd=f'cd ../ex-MoMo && micromamba run -n mdm python -m sample.transfer --model_path ../motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt --num_samples 3 --num_repetitions 3 {given_motion} --output_dir {output_path}/{now_date}_momo/{motion.replace(" ", "_")}/{personality_motion.replace(" ", "_")}'
            print(cmd)
            os.system(cmd)
            gc.collect()
            torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)