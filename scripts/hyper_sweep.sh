#!/bin/bash
# save this file as hyper_sweep.sh and give it execute permission with chmod +x hyper_sweep.sh

SWEEP_ID="lacc-lab/cognitive-state-decoding-condition-prediction/iezlibed" # Replace with actual sweep ID

# Check if tmux is installed
if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1
fi

# Start wandb agents in detached tmux sessions for each GPU
for i in 0 1 2 3; do
    session_name="wandb-gpu${i}"
    # Create a new detached tmux session and run the wandb agent command
    tmux new-session -d -s "$session_name" "CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID"
    echo "Launched W&B agent for GPU $i in tmux session $session_name."
done

echo "All W&B agents launched in separate tmux sessions. Reattach with: tmux attach-session -t <session_name>"
