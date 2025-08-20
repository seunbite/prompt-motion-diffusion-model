import yaml
import os
from tqdm import tqdm
import gc
import torch
from datetime import datetime
import glob
import multiprocessing as mp
from functools import partial
from utils import open_json, save_json


def process_chunk(chunk_data, gpu_id, output_path, now_date, arg, model_path, max_batch_size=1000):
    print(f"GPU {gpu_id}: Processing {len(chunk_data)} motions in batches of {max_batch_size}...")
    
    # Process in smaller batches to avoid memory issues
    total_batches = (len(chunk_data) + max_batch_size - 1) // max_batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * max_batch_size
        end_idx = min(start_idx + max_batch_size, len(chunk_data))
        batch_data = chunk_data[start_idx:end_idx]
        
        # Write batch to temporary file
        chunk_file = f'tmp_gpu_{gpu_id}_batch_{batch_idx}.txt'
        with open(chunk_file, 'w') as f:
            for p in batch_data:
                f.write(f'{p}\n')
        
        print(f"GPU {gpu_id}: Processing batch {batch_idx + 1}/{total_batches} ({len(batch_data)} motions)...")
        
        # Run the generation command with memory management
        # Use a unified output directory instead of separate batch directories
        unified_output_dir = f"{output_path}/{now_date}_mdm_{arg}/gpu_{gpu_id}"
        cmd = f"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 micromamba run -n mdm python -m sample.generate --model_path {model_path} --num_samples 3 --num_repetitions 1 --input_text {chunk_file} --output_dir {unified_output_dir}"
        
        # Execute with error handling
        exit_code = os.system(cmd)
        if exit_code != 0:
            print(f"GPU {gpu_id}: Error processing batch {batch_idx + 1}, exit code: {exit_code}")
        else:
            # Count generated files to show progress
            import glob
            motion_files = glob.glob(f"{unified_output_dir}/motion_*")
            print(f"GPU {gpu_id}: Successfully completed batch {batch_idx + 1}/{total_batches} (Total files: {len(motion_files)})")
        
        # Clean up temporary file
        os.remove(chunk_file)
        
        # Force garbage collection between batches
        import gc
        gc.collect()
    
    print(f"GPU {gpu_id}: Completed processing all {len(chunk_data)} motions")
    

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
    model_name_or_path: str = 'Qwen/Qwen2.5-VL-32B-Instruct',
    prompt_path: str = 'dataset/prompt',
    stage: str = 'prompt_momo', # gen_mdm, gen_momo, retarget_motion
    arg: str = 'type1',
    index_chunk: int = 2000,
    start: int = 0,
    max_model_len: int = 2048,
    max_num_seqs: int = 50,
    gpu_memory_utilization: float = 0.9,
    output_path: str = '/scratch2/iyy1112/motion-persona/save',
    retarget_timeout: int = 3000,
    retarget_gl: str = 'egl',
    max_batch_size: int = 1000,
    max_motions_per_gpu: int = 50000
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
        # prompt = """You are generating motion prompts with detail (e.g., A person walks with big bouncy steps) to reflect the given personality (e.g., high extraversion).
# Please provide 20 different motions prompts for each given personality.

# Steps:
# 1) Imagine a person with the given personality performing the given motion with diversity. (e.g, High extraversion - big bouncy steps, shaking head, shaking hands)
# 2) Describe the MOTION PROMPTS with ENHANCED DETAIL to clearly reflect the personality type. (Begin with the original [Motion], then add a description of the additional motion detail.)

# Constraints:
# - Don't specify the inner personality or mind state, just describe the **visible** motion detail.
# - For motion generation, the hand motion and face motion are not available.
# - Please keep the detailed motion instruction simple andclear with common motions.

# Format: 
# 1. A person walks with big bouncy steps
# 2. A person walks shaking hands
# 3. ...

# [Motion]: {}
# [Persona]: {} {}
# [Output]:
# 1.
# """
        prompt = """You are generating motion prompts with detail (e.g., A person walks with big bouncy steps) to reflect the given personality (e.g., high extraversion).
Please provide 20 different motions prompts for each given personality.

Steps:
1) If the given motion is a direct action (e.g., walk, run, jump), imagine a person with the given personality performing it with diversity. (e.g, High extraversion - big bouncy steps, shaking head, shaking hands)
2) If the given motion is a general behavior (e.g., mimicking a ghost, acting like a robot), break it down into specific, visible, and understandable motions that match the original behavior. Create diverse variations while maintaining consistency with the original action. (e.g., "mimicking a ghost" → "extends arms forward and walks with slow, tiptoeing steps", "sways body side to side while moving forward slowly", "crouches down and moves with bent knees in a sneaky manner")
3) Describe the MOTION PROMPTS with ENHANCED DETAIL to clearly reflect the personality type. (Begin with the original [Motion] concept, then add a description of the specific motion detail.)

Constraints:
- Don't specify the inner personality or mind state, just describe the **visible** motion detail.
- For motion generation, the hand motion and face motion are not available.
- Please keep the detailed motion instruction simple and clear with common motions.
- Break down abstract behaviors into concrete, executable body movements.

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
        source_verbs = open_json('save/motion_verb_score.cleaned.jsonl')
        for line in source_verbs:
                for personality in persons:
                    for hl in ['high', 'low']:
                        data.append({
                            'inputs': [line['motion'], hl.capitalize(), personality.lower()],
                            'hl': hl,
                            'personality': personality,
                            'motion': line['motion'],
                            **line
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
        prompts = [r['mdm_prompt'] for r in personality_with_motions]
        for video, prompt in zip(videos, prompts):
            data.append({
                'video_path': video,
                'inputs': [prompt] 
            })
            
        _ = get_results(
            model_name_or_path=model_name_or_path,
            prompt=prompt,
            data=data,
            max_tokens=1000,
            temperature=0.3,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            save_path=os.path.join(prompt_path, f'{now_date}_{stage}.jsonl')
        )
        
    elif stage == 'basic_captioning':
        from mylmeval.infer import get_results
        prompt = """Please generate a caption for the given video.

[Format]: A person do something.
[Output]:
"""
        data = []
        data_path = f"{output_path}/20250805_mdm_type3/*/*"
        videos = sorted(glob.glob(f"{data_path}/retarget_motion_*.mp4"))
        for video in videos:
            data.append({
                'original_video_path' : os.path.dirname(video),
                'original_prompt' : open(os.path.join(os.path.dirname(video), 'text.txt'), 'r').read(),
                'video_path': video,
                'inputs': [prompt] 
            })
            
        _ = get_results(
            model_name_or_path=model_name_or_path,
            prompt=prompt,
            data=data,
            batch_size=5,
            max_tokens=1000,
            temperature=0.1,
            max_num_seqs=5,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            save_path=os.path.join(prompt_path, f'{now_date}_{stage}.jsonl')
        )
        
    elif stage == 'gen_mdm':
        os.makedirs(os.path.join(output_path, f'{now_date}_mdm'), exist_ok=True)
        if arg == 'type2':
            with open(os.path.join('assets/dataseed/prompt_verb_100.txt'), 'r') as f:
                personalities = [p for p in f.readlines()]
            personalities = [p.strip() for p in personalities if len(p) > 3]
        elif arg == 'type3':
            data = open_json(os.path.join(prompt_path, '20250815_prompt_augmentation_direct.jsonl'))
            new_data = [
                [{**item, 'prompt' : r.strip(".").split(". ")[-1]} for r in item['result'].split("\n") if len(r) > 3 and '[' not in r]
                 for item in data]
            new_data = [p for sublist in new_data for p in sublist]
            save_json(new_data, os.path.join(prompt_path, f'prompts.jsonl'))
            personalities = [p['prompt'] for p in new_data]
        
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
        print(f"Each GPU will process motions in batches of {max_batch_size} to manage memory")
        
        # Create partial function with fixed arguments
        model_path = "./save/humanml_enc_512_50steps/model000750000.pt"
        worker_func = partial(process_chunk, 
                           output_path=output_path, 
                           now_date=now_date, 
                           arg=arg, 
                           model_path=model_path,
                           max_batch_size=max_batch_size)
        
        # Create processes for each GPU with error handling
        processes = []
        for gpu_id in range(num_gpus):
            if chunks[gpu_id]:  # Only create process if chunk is not empty
                try:
                    p = mp.Process(target=worker_func, args=(chunks[gpu_id], gpu_id))
                    processes.append(p)
                    p.start()
                    print(f"Started process for GPU {gpu_id}")
                except Exception as e:
                    print(f"Failed to start process for GPU {gpu_id}: {e}")
        
        # Wait for all processes to complete with progress bar
        print(f"Starting generation on {len(processes)} GPUs...")
        print(f"Monitor progress with: watch -n 5 'find {output_path}/{now_date}_mdm_{arg}/ -name \"motion_*\" | wc -l'")
        failed_processes = []
        with tqdm(total=len(processes), desc="GPU Processes", unit="GPU") as pbar:
            for i, p in enumerate(processes):
                try:
                    p.join()
                    if p.exitcode != 0:
                        print(f"Process {i} exited with code {p.exitcode}")
                        failed_processes.append(i)
                    else:
                        # Show final count for this GPU
                        import glob
                        gpu_files = glob.glob(f"{output_path}/{now_date}_mdm_{arg}/gpu_{i}/motion_*")
                        print(f"GPU {i} completed with {len(gpu_files)} motion files")
                    pbar.update(1)
                except KeyboardInterrupt:
                    print("Interrupted by user. Terminating processes...")
                    for proc in processes:
                        if proc.is_alive():
                            proc.terminate()
                    raise
                except Exception as e:
                    print(f"Error waiting for process {i}: {e}")
                    failed_processes.append(i)
                    pbar.update(1)
        
        if failed_processes:
            print(f"Warning: {len(failed_processes)} processes failed: {failed_processes}")
        else:
            print("All GPU processes completed successfully!")
        
        # Show final statistics
        import glob
        total_files = glob.glob(f"{output_path}/{now_date}_mdm_{arg}/*/motion_*")
        print(f"Total motion files generated: {len(total_files)}")
        print(f"Results saved in: {output_path}/{now_date}_mdm_{arg}/")
        
        # Show breakdown by GPU
        for gpu_id in range(len(processes)):
            gpu_files = glob.glob(f"{output_path}/{now_date}_mdm_{arg}/gpu_{gpu_id}/motion_*")
            if gpu_files:
                print(f"  GPU {gpu_id}: {len(gpu_files)} files")


    elif stage == 'retarget_motion':
        motion_files = glob.glob(f"{output_path}/20250805_mdm_type3/*/*/motion.npy")
        if not motion_files:
            print(f"No motion files found in {output_path}/20250805_mdm_type3/")
            return
        print(f"Found {len(motion_files)} motion files to retarget")
        motion_files = motion_files[start:start+index_chunk]
        
        # 이미 처리된 파일들 제외
        files_to_process = []
        for motion_file in motion_files:
            dirname = os.path.dirname(motion_file)
            motion_id = os.path.basename(dirname)
            output_file = os.path.join(dirname, f'retarget_{motion_id}.mp4')
            if not os.path.exists(output_file):
                files_to_process.append(motion_file)
        
        print(f"Files to process: {len(files_to_process)} (already completed: {len(motion_files) - len(files_to_process)})")
        
        if not files_to_process:
            print("All files already processed!")
            return
        
        def process_retarget_chunk(chunk_files, process_id, progress_queue):
            """Worker function to process a chunk of motion files"""
            completed_count = 0
            total_count = len(chunk_files)
            
            print(f"Process {process_id}: Starting {total_count} files...")
            
            for motion_file in chunk_files:
                try:
                    dirname = os.path.dirname(motion_file)
                    motion_id = os.path.basename(dirname)
                    output_file = os.path.join(dirname, f'retarget_{motion_id}.mp4')
                    
                    # Use subprocess with EGL for fast headless rendering
                    import subprocess, sys
                    cmd = [sys.executable, "-m", "sample.retarget_humanoid", 
                           "--motion_file", motion_file, "--output_file", output_file]
                    env = os.environ.copy()
                    env['MUJOCO_GL'] = retarget_gl
                    # Fallback to osmesa for CPU-only headless if EGL not available
                    if retarget_gl.lower() == 'egl' and torch.cuda.is_available():
                        num_devices = torch.cuda.device_count()
                        env['MUJOCO_EGL_DEVICE_ID'] = str(process_id % max(1, num_devices))
                    
                    try:
                        # 5-minute timeout per file
                        result = subprocess.run(cmd, text=True, env=env, timeout=retarget_timeout)
                        if result.returncode != 0:
                            print(f"Process {process_id}: Failed to retarget {motion_id}")
                        # If EGL fails due to driver, suggest switching to osmesa
                        if 'EGL' in str(result):
                            print("Hint: Try running with --retarget_gl=osmesa for CPU rendering.")
                        else:
                            completed_count += 1
                            progress_queue.put(('completed', process_id, motion_id))
                            print(f"Process {process_id}: Completed {completed_count}/{total_count} - {motion_id}")
                    except subprocess.TimeoutExpired:
                        print(f"Process {process_id}: Timeout (>5 min) on {motion_id}, skipping")
                        continue
                        
                except Exception as e:
                    print(f"Process {process_id}: Error processing {motion_file}: {e}")
                    continue
            
            # 완료 신호 전송
            progress_queue.put(('finished', process_id, completed_count))
            print(f"Process {process_id}: Finished with {completed_count}/{total_count} files")
        
        # Split motion files into chunks for CPU processes
        # num_processes = min(mp.cpu_count(), len(files_to_process))  # 파일 수보다 프로세스 수가 많으면 안됨
        num_processes = 8  # 파일 수보다 프로세스 수가 많으면 안됨
        chunk_size = max(1, len(files_to_process) // num_processes)
        chunks = []
        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_processes - 1 else len(files_to_process)
            chunks.append(files_to_process[start_idx:end_idx])
        
        print(f"Using {num_processes} CPU processes with {chunk_size} motion files per process")
        
        # 진행상황 추적을 위한 큐
        progress_queue = mp.Queue()
    
        
        # Create processes
        processes = []
        for process_id in range(num_processes):
            if chunks[process_id]:  # Only create process if chunk is not empty
                p = mp.Process(target=process_retarget_chunk, args=(chunks[process_id], process_id, progress_queue))
                processes.append(p)
                p.start()
        
        # 진행상황 모니터링
        print(f"Starting retargeting on {len(processes)} CPU processes...")
        total_files = len(files_to_process)
        completed_files = 0
        finished_processes = 0
        
        with tqdm(total=total_files, desc="Overall Progress", unit="file") as pbar:
            while finished_processes < len(processes):
                try:
                    # 큐에서 진행상황 메시지 수신 (타임아웃 설정)
                    message = progress_queue.get(timeout=1.0)
                    
                    if message[0] == 'completed':
                        completed_files += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'Completed': f"{completed_files}/{total_files}",
                            'Processes': f"{finished_processes}/{len(processes)}"
                        })
                        
                    elif message[0] == 'finished':
                        finished_processes += 1
                        pbar.set_postfix({
                            'Completed': f"{completed_files}/{total_files}",
                            'Processes': f"{finished_processes}/{len(processes)}"
                        })
                        
                except mp.queues.Empty:
                    # 타임아웃 - 진행상황 업데이트
                    pbar.set_postfix({
                        'Completed': f"{completed_files}/{total_files}",
                        'Processes': f"{finished_processes}/{len(processes)}"
                    })
                    continue
        
        print(f"All CPU processes completed! Total files processed: {completed_files}/{total_files}")

    elif stage == 'vlm_evaluation':
        from mylmeval.infer import get_results
        text_prompt = "What is this person doing? Describe it briefly, but if necessary, describe it in detail."
        data = []
        data_path = f"{output_path}/20250805_mdm_type3/*/*/retarget_motion_*.mp4"
        videos = sorted(glob.glob(data_path))
        print(f"Found {len(videos)} videos for VLM evaluation, chunk {start} to {start+index_chunk}")
        videos = videos[start:start+index_chunk]
        
        for video in videos:
            # Get the original prompt from text.txt file
            original_prompt_path = os.path.join(os.path.dirname(video), 'text.txt')
            original_prompt = ""
            if os.path.exists(original_prompt_path):
                try:
                    with open(original_prompt_path, 'r') as f:
                        original_prompt = f.read().strip()
                except Exception as e:
                    print(f"Error reading text.txt for {video}: {e}")
            
            data.append({
                'video_path': video,
                'original_video_path': os.path.dirname(video),
                'original_prompt': original_prompt,
                'inputs': [text_prompt] 
            })
            
        print(f"Processing {len(data)} videos with VLM evaluation")
        
        _ = get_results(
            model_name_or_path='google/gemma-3-27b-it',
            prompt=text_prompt,
            data=data,
            vlm=True,
            max_tokens=1000,
            save_path=os.path.join(prompt_path, f'{now_date}_{stage}.jsonl')
        )
        
        print(f"VLM evaluation completed. Results saved to {os.path.join(prompt_path, f'{now_date}_{stage}.jsonl')}")

    elif stage == 'gen_momo':
        motions = [r['motion'] for r in personality_with_motions]
        personalities = [r['personality'] for r in personality_with_motions]
        motion_personality_pairs = [(motion, personality) for motion in motions for personality in personalities]
        os.makedirs(os.path.join(output_path, 'momo'), exist_ok=True)
        for motion, personality in tqdm(motion_personality_pairs):
            given_motion = f'--text_leader "{motion}" --text_follower "{personality}"'
            cmd=f'cd ../ex-MoMo && micromamba run -n mdm python -m sample.transfer --model_path ../motion-diffusion-model/save/humanml_enc_512_50steps/model000750000.pt --num_samples 3 --num_repetitions 3 {given_motion} --output_dir {output_path}/{now_date}_momo/{motion.replace(" ", "_")}/{personality.replace(" ", "_")}'
            print(cmd)
            os.system(cmd)
            gc.collect()
            torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)