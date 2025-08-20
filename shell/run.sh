#!/bin/bash
source ~/.bashrc

arg1=${1:-"evolve"}
arg2=${1:-"retarget_motion"}
arg3=${2:-"test_tom"}
arg4=${3:-"test_tom"}

if [ $arg2 == "retarget_motion" ]; then
    env=mdm
elif [ $arg2 == "vlm_evaluation" ] || [ $arg2 == "prompt_augmentation" ]; then
    env=llm
else
    env=llm
fi

micromamba activate $env
cd ~/workspace/motion-diffusion-model


if [ $arg1 == "evolve" ]; then
    python evolve/evolve.py --stage ${arg2} --start ${arg3} --index_chunk ${arg4}
elif [ $arg1 == "motion_verb" ]; then
    python evolve/motion_verb.py
fi