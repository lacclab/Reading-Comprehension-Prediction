import os
import stat
from pathlib import Path
from typing import Literal

from tap import Tap


import wandb
from scripts.search_spaces_and_configs import run_configs, search_space_by_model_name


class HyperArgs(Tap):
    """
    Usage:

    1. check that 'search_space_by_model_name' has the correct hyperparameter search space for the model you wish to sweep.
    2. add a new 'RunConfig' to 'run_configs' with the desired hyperparameters.
    3. run 'python scripts/better_hyperparameters_sweep.py --config-name <config_name> --run-cap <run_cap> --wandb-project <wandb_project> --wandb-entity <wandb_entity>'
        * notes:
        * * 'config_name' should be the key of the 'RunConfig' in 'run_configs'.
        * * 'config_name' is the only required argument.
    3.cont. the script will create an executable bash script for each sweep (fold_idx), which will launch the wandb sweeps.
    4. run the bash script ./<bash_script>.sh
    5. input the GPU index and the number of runs you want to launch on that GPU to start the sweeps.
    """

    config_name: str  # Name of the config to use from running_configs.
    run_cap: int = 30  # Maximum number of runs to execute.
    wandb_project: str = "reading-comprehension-from-eye-movements"  # Name of the wandb project to log to.
    wandb_entity: str = "lacc-lab"  # Name of the wandb entity to log to.
    folds: list[int] = [0]  # List of fold indices to run.
    gpu_count: int = 1  # Number of GPUs to use.

    # Slurm settings.
    create_slurm: bool = False  # Create Slurm scripts.
    slurm_cpus: int = 12  # Number of CPUs to use. Ideally number of workers + 2.
    slurm_mem: str = "64G"  # Amount of memory to use.
    search_algorithm: Literal["bayes", "grid", "random"] = (
        "grid"  # Search algorithm to use.
    )


def create_sweep_configs(hyper_args):
    cfg = run_configs[hyper_args.config_name]
    search_space = search_space_by_model_name[cfg.model_name]
    return [
        {
            "program": "src/train.py",
            "method": hyper_args.search_algorithm,
            "metric": {
                "goal": "maximize",
                "name": "Balanced_Accuracy/val_best_epoch_weighted_average",
            },
            "run_cap": hyper_args.run_cap,
            "entity": hyper_args.wandb_entity,
            "project": hyper_args.wandb_project,
            "parameters": search_space,
            "command": [
                "${env}",
                "${interpreter}",
                "${program}",
                "${args_no_hypens}",
                f"+model={cfg.model_variant}",
                f"+data={cfg.data_variant}",
                f"+data_path={cfg.data_path}",
                f"+trainer={cfg.trainer_variant}",
                f"data.fold_index={fold_idx}",
                f"trainer.devices={hyper_args.gpu_count}",
                f"trainer.wandb_job_type=hyperparameter_sweep_{cfg.model_variant}",
            ],
        }
        for fold_idx in hyper_args.folds
    ]


def launch_sweeps(hyper_args, sweep_configs):
    sweep_ids = [
        wandb.sweep(
            cfg, entity=hyper_args.wandb_entity, project=hyper_args.wandb_project
        )
        for cfg in sweep_configs
    ]
    return sweep_ids


def create_bash_scripts(hyper_args, sweep_ids):
    # time_rn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = Path("sweep_configs")
    base_path.mkdir(parents=True, exist_ok=True)
    for fold_index, sweep_id in zip(hyper_args.folds, sweep_ids):
        filename = (
            base_path / f"sweep_{hyper_args.config_name}_{fold_index}.sh"
        )  # .time={time_rn}
        with open(filename, "w") as f:
            f.write(
                f"""#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source $HOME/miniforge3/etc/profile.d/conda.sh
source $HOME/miniforge3/etc/profile.d/mamba.sh
mamba activate decoding
cd $HOME/Cognitive-State-Decoding
git checkout emnlp24-postsubmission
git pull

GPU_NUM=$1
RUNS_ON_GPU=${{2:-1}}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${{GPU_NUM}}-dup${{i}}-{sweep_id}"
    tmux new-session -d -s "${{session_name}}" "CUDA_VISIBLE_DEVICES=${{GPU_NUM}} wandb agent {hyper_args.wandb_entity}/{hyper_args.wandb_project}/{sweep_id}"; tmux set-option -t "${{session_name}}" remain-on-exit on
    echo "Launched W&B agent for GPU ${{GPU_NUM}}, Dup ${{i}} in tmux session ${{session_name}}"
    sleep 2
done
"""
            )
        os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)
        print(f"Created bash script: {filename}")


def create_slurm_scripts(hyper_args: HyperArgs, sweep_ids: list[str], slurm_qos: str):
    extension = {
        "PostFusionAnswersH": "PFHA",
        "PostFusionMultiClassH": "PFHM",
        "PostFusionAnswersMultiClassH": "PFHAM",
        "PostFusionAnswersG": "PFGA",
        "PostFusionMultiClassG": "PFGM",
        "PostFusionAnswersMultiClassG": "PFGAM",
        "PostFusionSelectedAnswersMultiClassG": "PFSG",
        "PostFusionSelectedAnswersMultiClassH": "PFSH",
        "RobertaSelectedAnswersMultiClassH": "RSH",
        "RobertaSelectedAnswersMultiClassG": "RSG",
        "MAGFreezeH": "MFH",
        "MAGFreezeG": "MFG",
        "MAGBaseH": "MBH",
        "MAGBaseG": "MBG",
        "MAGEyesH": "MEH",
        "MAGEyesG": "MEG",
        "MAGWordsH": "MWH",
        "MAGWordsG": "MWG",
        "PostFusionFreezeH": "PFFH",
        "PostFusionFreezeG": "PFFG",
        "RoberteyeWordSelectedAnswersMultiClassH": "RWH",
        "RoberteyeWordSelectedAnswersMultiClassG": "RWG",
        "RoberteyeWordLingSelectedAnswersMultiClassH": "RWLH",
        "RoberteyeWordLingSelectedAnswersMultiClassG": "RWLG",
        "RoberteyeFixationSelectedAnswersMultiClassH": "RFH",
        "RoberteyeFixationSelectedAnswersMultiClassG": "RFG",
    }

    filename = f"slurm_sweep_{hyper_args.config_name}_{slurm_qos}.job"

    with open(filename, "w") as f:
        f.write(
            f"""#!/bin/bash

#SBATCH --job-name={extension.get(hyper_args.config_name, hyper_args.config_name)}-array
#SBATCH --output=slurm_log/slurm-{extension.get(hyper_args.config_name, hyper_args.config_name)}-%A_%a.out
#SBATCH --partition=work,mig
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --gpus={hyper_args.gpu_count}
#SBATCH --qos={slurm_qos}
#SBATCH --cpus-per-task={hyper_args.slurm_cpus}
#SBATCH --mem={hyper_args.slurm_mem}
#SBATCH --array=0-{len(sweep_ids)-1}

sweep_ids=({' '.join(sweep_ids)})

srun --overlap --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK -p work,mig wandb agent "{hyper_args.wandb_entity}/{hyper_args.wandb_project}/${{sweep_ids[$SLURM_ARRAY_TASK_ID]}}" &
sleep 600
srun --overlap --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK -p work,mig wandb agent "{hyper_args.wandb_entity}/{hyper_args.wandb_project}/${{sweep_ids[$SLURM_ARRAY_TASK_ID]}}" &
sleep 10
srun --overlap --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK -p work,mig wandb agent "{hyper_args.wandb_entity}/{hyper_args.wandb_project}/${{sweep_ids[$SLURM_ARRAY_TASK_ID]}}"
wait
"""
        )

    os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)
    print(f"Created Slurm array job script: {filename}")


def main():
    hyper_args = HyperArgs().parse_args()

    print(f"Hyper Args:\n{hyper_args}")

    sweep_configs = create_sweep_configs(hyper_args)
    sweep_ids = launch_sweeps(hyper_args, sweep_configs)

    if hyper_args.create_slurm:
        create_slurm_scripts(hyper_args, sweep_ids, slurm_qos="normal")
        create_slurm_scripts(hyper_args, sweep_ids, slurm_qos="basic")
    else:
        create_bash_scripts(hyper_args, sweep_ids)

    # save sweep ids to a file together with the hyper_args
    with open(f"sweep_ids_{hyper_args.config_name}.txt", "w") as f:
        f.write(f"Hyper Args:\n{hyper_args}\n")
        f.write(f"Sweep IDs:\n{sweep_ids}")


if __name__ == "__main__":
    main()
