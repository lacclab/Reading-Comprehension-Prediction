user=$(whoami)

paths=(
    # "outputs/+data=Hunting,+data_path=may05,+model=RoberteyeFixation,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeFixation"
    # "outputs/+data=Hunting,+data_path=may05,+model=PostFusion,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusion"
    # "outputs/+data=Hunting,+data_path=may05,+model=MAG,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_MAG"
    # "outputs/+data=Hunting,+data_path=may05,+model=Roberta,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_Roberta"
    # "outputs/+data=Hunting,+data_path=may05,+model=RoberteyeWord,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeWord"
    # "outputs/+data=Gathering,+data_path=may05,+model=RoberteyeWord,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeWord"
    # "outputs/+data=Gathering,+data_path=may05,+model=RoberteyeFixation,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeFixation"
    # "outputs/+data=Gathering,+data_path=may05,+model=PostFusion,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusion"
    # "outputs/+data=Gathering,+data_path=may05,+model=MAG,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_MAG"
    # "outputs/+data=Gathering,+data_path=may05,+model=Roberta,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_Roberta"
    # "outputs/+data=Gathering,+data_path=may05,+model=BEyeLSTMArgs,+trainer=BEyeLSTM,trainer.wandb_job_type=hyperparameter_sweep_BEyeLSTMArgs"
    # "outputs/+data=Hunting,+data_path=may05,+model=BEyeLSTMArgs,+trainer=BEyeLSTM,trainer.wandb_job_type=hyperparameter_sweep_BEyeLSTMArgs"
    # "outputs/+data=Gathering,+data_path=may05,+model=PostFusionAnswers,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusionAnswers"
    # "outputs/+data=Hunting,+data_path=may05,+model=PostFusionAnswers,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusionAnswers"
    # "outputs/+data=Gathering,+data_path=may05,+model=PostFusionMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusionMultiClass"
    # "outputs/+data=Hunting,+data_path=may05,+model=PostFusionMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusionMultiClass"
    "outputs/+data=Gathering,+data_path=may05,+model=RobertaSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RobertaSelectedAnswersMultiClass"
    "outputs/+data=Hunting,+data_path=may05,+model=RobertaSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RobertaSelectedAnswersMultiClass"
    "outputs/+data=Gathering,+data_path=may05,+model=PostFusionSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusionSelectedAnswersMultiClass"
    "outputs/+data=Hunting,+data_path=may05,+model=PostFusionSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusionSelectedAnswersMultiClass"
    "old_outputs/+data=Gathering,+data_path=may05,+model=RobertaSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RobertaSelectedAnswersMultiClass"
    "old_outputs/+data=Hunting,+data_path=may05,+model=RobertaSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RobertaSelectedAnswersMultiClass"
    "old_outputs/+data=Gathering,+data_path=may05,+model=PostFusionSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusionSelectedAnswersMultiClass"
    "old_outputs/+data=Hunting,+data_path=may05,+model=PostFusionSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusionSelectedAnswersMultiClass"
    "outputs/+data=Gathering,+data_path=may05,+model=MAGSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_MAGSelectedAnswersMultiClass"
    "outputs/+data=Hunting,+data_path=may05,+model=MAGSelectedAnswersMultiClass,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_MAGSelectedAnswersMultiClass"
)

while IFS= read -r server; do
    # Skip if server starts with #
    if [[ $server == \#* ]]; then
        echo "Skipping server: $server"
        continue
    fi
    echo "Synchronizing data with $server"
    for path in "${paths[@]}"; do
        rsync -avzP --mkpath --append-verify $user@$server.iem.technion.ac.il:Cognitive-State-Decoding/${path}/ emnlp24_outputs/${path}
        done
done <utils/server_sync/server_list.txt


# rsync -avzP --append-verify $user@dgx-master.technion.ac.il:/rg/berzak_prj/$user/Cognitive-State-Decoding/${path}/ emnlp24_outputs/${path}
