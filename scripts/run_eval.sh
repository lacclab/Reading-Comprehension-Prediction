paths=(
"synced_outputs/+data=Hunting,+data_path=may05,+model=RoberteyeFixation,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeFixation"
"synced_outputs/+data=Hunting,+data_path=may05,+model=RoberteyeWord,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeWord"
"synced_outputs/+data=Hunting,+data_path=may05,+model=PostFusion,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusion"
"synced_outputs/+data=Hunting,+data_path=may05,+model=MAG,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_MAG"
"synced_outputs/+data=Hunting,+data_path=may05,+model=Roberta,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_Roberta"
"synced_outputs/+data=Gathering,+data_path=may05,+model=RoberteyeFixation,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeFixation"
"synced_outputs/+data=Gathering,+data_path=may05,+model=RoberteyeWord,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_RoberteyeWord"
"synced_outputs/+data=Gathering,+data_path=may05,+model=PostFusion,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_PostFusion"
"synced_outputs/+data=Gathering,+data_path=may05,+model=MAG,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_MAG"
"synced_outputs/+data=Gathering,+data_path=may05,+model=Roberta,+trainer=IsCorrectSampling,trainer.wandb_job_type=hyperparameter_sweep_Roberta"
)

for path in "${paths[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/eval.py "eval_path=\"${path}\""
done
