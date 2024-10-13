#!/bin/bash
search_algorithm=grid
folds="0 1 2 3 4 5 6 7 8 9" # 1 2 3 4 5 6 7 8 9" # separate by comma, e.g. "0,1,2,3,4"
wandb_project="emnlp24-postsubmission"
slurm=false
config_names=(
    # "RoberteyeFixationG"
    # "RoberteyeFixationH"
    # "RoberteyeWordG"
    # "RoberteyeWordH"
    # "PostFusionG"
    # "PostFusionH"
    # "MAGG"
    # "MAGH"
    # "RobertaH"
    # "RobertaG"
    # "BEyeLSTMG"
    # "BEyeLSTMH"
    # "AhnCNNG"
    # "AhnCNNH"
    # "EyettentionG"
    # "EyettentionH"
    # "PostFusionAnswersH"
    # "PostFusionAnswersG"
    # "PostFusionMultiClassH"
    # "PostFusionMultiClassG"
    # "PostFusionAnswersMultiClassH"
    # "PostFusionAnswersMultiClassG"
    # "PostFusionSelectedAnswersMultiClassH"
    # "PostFusionSelectedAnswersMultiClassG"
    # "RobertaSelectedAnswersMultiClassH"
    # "RobertaSelectedAnswersMultiClassG"
    # "MAGSelectedAnswersMultiClassH"
    # "MAGSelectedAnswersMultiClassG"
    # "MAGRaceH"
    # "MAGRaceG"
    # "MAGBaseH"
    # "MAGBaseG"
    # "MAGFreezeH"
    # "MAGFreezeG"
    # "MAGEyesH"
    # "MAGEyesG"
    # "MAGWordsH"
    # "MAGWordsG"
    # "PostFusionFreezeH"
    # "PostFusionFreezeG"
    # "RoberteyeWordLingG"
    # "RoberteyeWordLingH"
    # "MAGSelectedAnswersMultiClassLingG"
    # "MAGSelectedAnswersMultiClassLingH"
    # "RoberteyeWordSelectedAnswersMultiClassH"
    # "RoberteyeWordSelectedAnswersMultiClassG"
    # "RoberteyeWordLingSelectedAnswersMultiClassH"
    # "RoberteyeWordLingSelectedAnswersMultiClassG"
    # "RoberteyeFixationSelectedAnswersMultiClassH"
    # "RoberteyeFixationSelectedAnswersMultiClassG"
    "RoberteyeWordEyesH"
    "RoberteyeWordEyesG"
)
run_cap=200
slurm_cpus=9
slurm_mem="75G"
for config_name in "${config_names[@]}"; do
    if [ "$slurm" = true ]; then
        python scripts/better_hyperparameters_sweep.py --config_name=$config_name --search_algorithm=$search_algorithm --folds $folds --wandb_project=$wandb_project --run_cap=$run_cap --create_slurm --slurm_cpus=$slurm_cpus --slurm_mem=$slurm_mem
    else
        python scripts/better_hyperparameters_sweep.py --config_name=$config_name --search_algorithm=$search_algorithm --folds $folds --wandb_project=$wandb_project --run_cap=$run_cap
    fi
done
