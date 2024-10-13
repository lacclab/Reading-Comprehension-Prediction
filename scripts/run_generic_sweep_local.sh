#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi
user=$(whoami)
export PYTHONPATH=$PYTHONPATH:/data/home/$user/Cognitive-State-Decoding                                                                                                   
source $HOME/miniforge3/etc/profile.d/conda.sh
source $HOME/miniforge3/etc/profile.d/mamba.sh
mamba activate decoding
cd $HOME/Cognitive-State-Decoding
# git checkout emnlp24-postsubmission
# git pull

sweep_id=$1
GPU_NUM=$2
RUNS_ON_GPU=${3:-1}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${GPU_NUM}-dup${i}-${sweep_id}"
    tmux new-session -d -s "${session_name}" "export PYTHONPATH=$PYTHONPATH:/data/home/$user/Cognitive-State-Decoding; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent lacc-lab/emnlp24-postsubmission/${sweep_id}"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
    sleep 2
done
