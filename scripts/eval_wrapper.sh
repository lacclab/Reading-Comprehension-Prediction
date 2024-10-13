data=(
    "Hunting"
    "Gathering"
)
models=(
    #"RoberteyeFixation"
    #"RoberteyeWord"
    #"PostFusion"
    "MAG"
    #"Roberta"
    # "BEyeLSTM"
    # "AhnCNN"
    # "MAGRace"
    # "MAGBase"
    # "MAGFreeze"
    # "MAGEyes"
    # "MAGWords"
    # "PostFusionFreeze"
    )

trainer=IsCorrectSampling
data_path=may05
base_path=synced_outputs
{
    printf "paths=(\n"
    for data_item in "${data[@]}"; do
        for model in "${models[@]}"; do
            printf "\"%s/+data=%s,+data_path=%s,+model=%s,+trainer=%s,trainer.wandb_job_type=hyperparameter_sweep_%s\"\n" "$base_path" "$data_item" "$data_path" "$model" "$trainer" "$model"
        done
    done
    printf ")\n\n"
    printf "for path in \"\${paths[@]}\"; do\n"
    printf "    CUDA_VISIBLE_DEVICES=0 python src/eval.py \"eval_path=\\\\\"\${path}\\\\\"\"\n"
    printf "done\n"
} >run_eval_appendix.sh
