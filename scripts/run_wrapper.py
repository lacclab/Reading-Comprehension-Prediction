import shutil
import subprocess
from itertools import product
from pathlib import Path
from time import sleep
from typing import Literal

import torch
from tap import Tap

#! Write EXACTLY as in the registered configs!!!
TrainerOptions = Literal[
    "Shubi",
    "ReadingCompBase",
    "IsCorrectSampling",
    "Ahn",
    "BEyeLSTM",
    "Yoav",
    "Eyettention",
    "CfirRoBERTaEye",
    "CfirBEyeLSTM",
    "CfirMAG",
]
DataOptions = Literal[
    "DataArgs",
    "NoReread",
    "Hunting",
    "Gathering",
    "HuntingCSOnly",
    "DataPathArgs",
    "feb11",
]
ModelOptions = Literal[
    "AhnArgs",
    "AhnRNN",
    "AhnCNN",
    "BEyeLSTMArgs",
    "BEyeLSTMQCond",
    "BEyeLSTMIsCorrectCLSampling",
    "BEyeLSTMIsCorrectSampling",
    "EyettentionArgs",
    "FSEArgs",
    "LitArgs",
    "MAGArgs",
    "MAGQPredConcatNoFix",
    "MAGQPredDuplicateNoFix",
    "MAGQCondPredConcatNoFix",
    "MAGQCondPredDuplicateNoFix",
    "MAGCondPredConcatNoFix",
    "PostFusionArgs",
    "RoBERTeyeArgs",
    "RoBERTeyeDuplicate",
    "RoBERTeyeDuplicateFixation",
    "RoBERTaNoEyes",
    "RoBERTeyeConcat",
    "RoBERTeyeCondition",
    "RoBERTeyeQCondConcat",
    "RoBERTeyeQCondDuplicate",
    "RoBERTeyeCondPredConcat",
    "RoBERTeyeQPredDuplicateNoFix",
    "RoBERTeyeQPredConcatNoFix",
    "RoBERTeyeConcatFixationReadingComp",
    "RoBERTeyeConcatIAReadingComp",
    "MAGConcatReadingComp",
    "MLPArgs",
    "PostFusionReadingComp",
    "RoBERTeyeConcatIAIsCorrect",
    "RoBERTeyeConcatIAIsCorrectCL",
    "RoBERTeyeConcatIAIsCorrectCLSampling",
    "RoBERTeyeConcatFixationIsCorrectCLSampling",
    "RoBERTeyeConcatNoEyesIsCorrect",
    "PostFusionConcatIsCorrectCLSampling",
    "MAGConcatIsCorrectCLSampling",
    "RoBERTeyeConcatNoEyesIsCorrectCLSampling",
    "RoberteyeWord",
    "RoberteyeFixation",
    "PostFusion",
    "MAG",
    "Roberta",
    "PostFusionAnswers",
    "PostFusionMultiClass",
    "PostFusionAnswersMultiClass",
    "PostFusionNoLinguistic",
    "PostFusionSelectedAnswersMultiClass",
    "RobertaSelectedAnswersMultiClass",
    "MAGSelectedAnswersMultiClass",
    "MAGBase",
    "MAGFreeze",
    "MAGEyes",
    "MAGWords",
    "PostFusionFreeze",
    "RoberteyeWordLing",
    "MAGSelectedAnswersMultiClassLing",
    "RoberteyeWordSelectedAnswersMultiClass",
    "RoberteyeWordLingSelectedAnswersMultiClass",
    "RoberteyeFixationSelectedAnswersMultiClass",
    
]
DataPathOptions = Literal["april14", "may05"]

# ML
MLTrainerOptions = Literal["CfirMLQCond", "CfirMLQPred", "CfirJunk"]
MLModelOptions = Literal[
    # ! Question Prediction
    "LogisticRegressionQPredMLArgs",
    "KNearestNeighborsQPredMLArgs",
    "SupportVectorMachineQPredMLArgs",
    "DummyClassifierQPredMLArgs",
    # ! Condition Prediction
    # Logistic Regression
    "LogisticRegressionCondPredDavidILFMLArgs",
    "LogisticRegressionCondPredLennaILFMLArgs",
    "LogisticRegressionCondPredDianeILFMLArgs",
    "LogisticRegressionCondPredReadingTimeMLArgs",
    # Dummy Classifier
    "DummyClassifierCondPredMLArgs",
    # K-Nearest Neighbors
    "KNearestNeighborsCondPredDavidILFMLArgs",
    "KNearestNeighborsCondPredLennaILFMLArgs",
    "KNearestNeighborsCondPredDianeILFMLArgs",
    "KNearestNeighborsCondPredReadingTimeMLArgs",
    # Support Vector Machine
    "SupportVectorMachineCondPredDavidILFMLArgs",
    "SupportVectorMachineCondPredLennaILFMLArgs",
    "SupportVectorMachineCondPredDianeILFMLArgs",
    "SupportVectorMachineCondPredReadingTimeILFMLArgs",
    # ! IsCorrect Prediction
    # Logistic Regression
    "LogisticRegressionIsCorrectPredDavidILFMLArgs",
    "LogisticRegressionIsCorrectPredLennaILFMLArgs",
    "LogisticRegressionIsCorrectPredDianeILFMLArgs",
    "LogisticRegressionIsCorrectPredReadingTimeMLArgs",
    # Dummy Classifier
    "DummyClassifierIsCorrectPredMLArgs",
    # K-Nearest Neighbors
    "KNearestNeighborsIsCorrectPredDavidILFMLArgs",
    "KNearestNeighborsIsCorrectPredLennaILFMLArgs",
    "KNearestNeighborsIsCorrectPredDianeILFMLArgs",
    "KNearestNeighborsIsCorrectPredReadingTimeMLArgs",
    # Support Vector Machine
    "SupportVectorMachineIsCorrectPredDavidILFMLArgs",
    "SupportVectorMachineIsCorrectPredLennaILFMLArgs",
    "SupportVectorMachineIsCorrectPredDianeILFMLArgs",
    "SupportVectorMachineIsCorrectPredReadingTimeILFMLArgs",
]


class ArgParser(Tap):
    single_run: bool = True  # Flag for multi-run
    sweep: bool = False  # Flag for hyperparameter sweep, only used for eval for now

    trainer: TrainerOptions = "Shubi"
    data_options: list[DataOptions] = ["Hunting"]
    model_options: list[ModelOptions] = ["RoBERTeyeDuplicate"]

    do_not_keep_pane_alive: bool = (
        False  # False = Keep tmux pane alive after run ends (useful for debugging)
    )
    skip_train: bool = False  # Skip training and only run eval
    skip_eval: bool = False  # Skip eval and only run training

    # Hydra's parser considers some characters to be illegal in unquoted strings.
    # These otherwise special characters may be included in unquoted values by escaping them with a \.
    #! These characters are: \()[]{}:=, \t (the last two ones being the whitespace and tab characters).
    # See https://hydra.cc/docs/1.2/advanced/override_grammar/basic/#escaped-characters-in-unquoted-values
    other_params: list[str] = [
        # "model.use_fixation_report=True",
        # r"trainer.max_epochs\=1",
        # r"trainer.wandb_job_type\=hyperparam_sweep_post_fusion",
        r"trainer.wandb_job_type\=try_out_contrastive_loss",
        # "data.normalization_type=Scaler.MIN_MAX_SCALER"
    ]  # Other params to pass to train.py and eval.py
    dry_run: bool = False  # Dry run to get the list of runs that will be launched
    gpu_skip_list: list[int] | None = (
        None  # list of GPUs to skip if you know they are being used
    )
    gpu_force_list: list[int] | None = (
        None  # list of GPUs to force if you know they are not being used
    )

    # make sure not both are set to True
    def process_args(self) -> None:
        assert not (
            self.skip_train and self.skip_eval
        ), "Cannot skip both training and evaluation!"


def main():
    if not is_tmux_installed():
        print("tmux could not be found, please install tmux first.")
        exit(1)

    args = ArgParser().parse_args()

    trainer = args.trainer
    data_options = args.data_options
    model_options = args.model_options

    n_runs = len(data_options) * len(model_options)
    gpus = get_gpus(
        n_runs=n_runs, force_list=args.gpu_force_list, skip_list=args.gpu_skip_list
    )

    print(f"\nAvailable GPUs: {gpus}")
    base_eval_path, multi_run, fold_index = get_info_based_on_mode(
        single_run=args.single_run, sweep=args.sweep
    )
    print(f"\nLaunching the following runs with {args.do_not_keep_pane_alive=}:")

    Path(base_eval_path).mkdir(parents=True, exist_ok=True)

    sessions = {}
    model_data_combinations = product(data_options, model_options)
    for gpu, (data, model) in zip(gpus, model_data_combinations):
        session_name = f"gpu{gpu}-{data}-{model}"
        print(
            f"\n{session_name} - GPU: {gpu}, Data: {data}, Model: {model}, Trainer: {trainer}\n"
        )

        train_command = rf"CUDA_VISIBLE_DEVICES=${gpu} python src/train.py {multi_run} +data={data} data.fold_index={fold_index} +model={model} +trainer={trainer}"
        eval_command = rf"CUDA_VISIBLE_DEVICES=${gpu} python src/eval.py 'eval_path={base_eval_path}/+data\={data}\,+model\={model}\,+trainer\={trainer}"
        if args.other_params:
            print(f"Other params: {args.other_params}")
            train_command += " " + " ".join(args.other_params)
            eval_command += r"\," + r"\,".join(args.other_params)

        full_command = r"wandb online"
        if not args.skip_train:
            full_command += rf" && {train_command}"
        if not args.skip_eval:
            full_command += rf" && {eval_command}'"

        exit_if_tmux_pane_exists(session_name)

        sessions[session_name] = full_command

    print("Runs that will have been launched:")
    for session_name, full_command in sessions.items():
        print(f"\n{session_name}: {full_command}")

    if args.dry_run:
        print("\nDry run complete. No runs were launched.")
        exit(0)

    input("Press ENTER to confirm or CTRL+C to cancel.")

    for session_name, full_command in sessions.items():
        print(f"Launching: {full_command}")
        create_tmux_session(session_name, full_command, args.do_not_keep_pane_alive)
        print(f"Launched! Reattach with: tmux a -t {session_name}\n")
        sleep(2)

    print("All runs launched in separate tmux sessions. tmux ls:")
    subprocess.run("tmux ls", shell=True, check=True)


def get_info_based_on_mode(
    single_run=False,
    sweep=False,
) -> tuple[
    Literal["cross_validation_runs", "outputs"],
    Literal["-m", ""],
    Literal["0,2,4,6,8", "0"],
]:
    #! Must match config.config!
    if sweep:
        base_eval_path = "outputs"
    elif single_run:
        base_eval_path = "outputs"
    else:
        base_eval_path = "cross_validation_runs"

    multi_run = (not single_run) or sweep
    multi_run = "-m" if multi_run else ""
    fold_index = "0,2,4,6,8" if multi_run else "0"

    print(
        f"\nSingle-run flag: {single_run}. Folds = {fold_index} and results base path set to: {base_eval_path}"
    )

    return base_eval_path, multi_run, fold_index


def exit_if_tmux_pane_exists(session_name: str) -> None:
    """
    Exit the program if a tmux session with the given name exists.

    Args:
        session_name (str): The name of the tmux session.
    """
    result = subprocess.run(
        f"tmux list-panes -t {session_name}", shell=True, capture_output=True
    )

    if result.returncode == 0:
        print(
            f'"{session_name}" already exists! Delete it with "tmux kill-session -t {session_name}" if you want to re-run it. Exiting.'
        )
        exit(1)


def is_tmux_installed() -> bool:
    return shutil.which("tmux") is not None


def create_tmux_session(session_name, command, do_not_keep_pane_alive) -> None:
    base_command = rf'tmux new -d -s "{session_name}" "{command}"'
    if not do_not_keep_pane_alive:
        base_command += rf'\; set-option -t "{session_name}" remain-on-exit on'
    subprocess.run(base_command, shell=True, check=True)


def get_gpus(
    n_runs: int,
    force_list: list[int] | None = None,
    skip_list: list[int] | None = None,
) -> list[int]:
    """
    Get available GPUs with 0% utilization, respecting user-defined skipping and forcing options.

    Args:
        data_options (list or tuple): list or tuple of data options.
        model_options (list or tuple): list or tuple of model options.
        force_list (list[int], optional): list of GPU IDs to force the use of. Defaults to None.
        skip_list (list[int], optional): list of GPU IDs to skip. Defaults to None.

    Returns:
        list[int]: list of available GPU device IDs.
    """

    # Get available GPUs (with 0% utilization)
    gpus = [
        device_id
        for device_id in range(torch.cuda.device_count())
        if torch.cuda.utilization(device_id) == 0
    ]

    if skip_list is not None:
        gpus = [gpu for gpu in gpus if gpu not in skip_list]

    gpus = force_list if force_list is not None else gpus

    if len(gpus) < n_runs:
        raise ValueError(
            f"Number of GPUs must match number of runs! Got GPUs {gpus} but {n_runs=}."
        )

    return gpus[:n_runs]


if __name__ == "__main__":
    main()
