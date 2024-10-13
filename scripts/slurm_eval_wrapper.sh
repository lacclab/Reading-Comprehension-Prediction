#!/bin/bash

data=(
    "Hunting"
    "Gathering"
)
models=(
    #"RoberteyeFixation"
    #"RoberteyeWord"
    #"PostFusion"
    #"MAG"
    #"Roberta"
    # "BEyeLSTM"
    # "AhnCNN"
    "MAGRace"
    "MAGBase"
    "MAGFreeze"
    "MAGEyes"
    "MAGWords"
    "PostFusionFreeze"
)

trainer=IsCorrectSampling
data_path=may05
base_path=outputs
experiment_name=words_and_backbone
for data_item in "${data[@]}"; do
    for model in "${models[@]}"; do
        job_name="${data_item}${model}"
        output="slurm_log/slurm-eval-${job_name}-%j.out"
        sbatch_file="eval_${experiment_name}_${job_name}.job"

        {
            echo "#!/bin/bash"
            echo "#SBATCH --job-name=${job_name}"
            echo "#SBATCH --output=${output}"
            echo "#SBATCH --partition=work"
            echo "#SBATCH --ntasks=1"
            echo "#SBATCH --nodes=1"
            echo "#SBATCH --gpus=1"
            echo "#SBATCH --qos=normal"
            echo "#SBATCH --cpus-per-task=12"
            echo "#SBATCH --mem=50G"
            echo ""
            echo "srun --ntasks=1 --nodes=1 --cpus-per-task=\$SLURM_CPUS_PER_TASK -p work python src/eval.py \"eval_path=\\\"${base_path}/+data=${data_item},+data_path=${data_path},+model=${model},+trainer=${trainer},trainer.wandb_job_type=hyperparameter_sweep_${model}\\\"\""
        } >"${sbatch_file}"
        chmod +x "${sbatch_file}"
        # sbatch "${sbatch_file}"
    done
done
