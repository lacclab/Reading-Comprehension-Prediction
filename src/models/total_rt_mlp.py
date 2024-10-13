"""Fixation Sequence Encoder Based Model class (inherits from BaseModel)"""
# from src.configs.constants import (
#     MAX_SCANPATH_LENGTH,
# )
# from src.configs.enums import PredictionMode

import torch
from torch import nn

from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.base_model_args import BaseModelArgs
from src.configs.trainer_args import Base
from src.models.base_model import BaseModel

# from transformers import RobertaModel
# import transformers
# import warnings


class TotalRtMLP(BaseModel):
    """
    TODO add docstring
    """

    def __init__(
        self,
        model_args: BaseModelArgs,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(model_args, trainer_args)

        self.model_args = model_args
        self.input_size = model_args.eyes_dim
        self.hidden_size = model_args.model_params.hidden_dim

        num_classes = len(model_args.prediction_config.class_names)
        self.output_size = num_classes if num_classes > 2 else 1

        self.max_len = model_args.max_seq_len * self.input_size

        if trainer_args.accelerator == "auto":
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        elif trainer_args.accelerator == "cpu":
            self.device_ = "cpu"
        elif trainer_args.accelerator == "gpu":
            self.device_ = "cuda"

        # fc_dropout = model_args.fc_dropout if model_args.fc_dropout else 0.3
        self.lr_fc = nn.Linear(1, 1).to(self.device_)
        self.bn = nn.BatchNorm1d(1).to(self.device_)
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # batch normalization
        x = self.bn(x)
        x = self.lr_fc(x)
        return x

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_dict = super().unpack_batch(batch)

        fix_rep_features = (
            self.model_args.fixation_features + self.model_args.ia_features
        )

        p_total_rt_idx = fix_rep_features.index("PARAGRAPH_RT")

        p_total_rts = batch_dict["fixations_features"][:, :, p_total_rt_idx]
        # take the mean of current_fixations_durations while ignoring zeros
        p_total_rts = torch.where(
            p_total_rts == 0,
            torch.tensor(float("nan")),
            p_total_rts,
        )
        p_total_rts = torch.nanmean(p_total_rts, dim=1)

        p_tokenized_lengths = batch_dict["p_input_masks"].sum(dim=1)

        normalized_total_p_rt = (p_total_rts / p_tokenized_lengths).unsqueeze(1)

        label = batch_dict.labels

        # Inversions: For each paragraph a list, where the value at entry i is the
        # IA_ID that is associated with the i-thtoken in the paragraph (batch_size, MAX_SEQ_LENGTH)
        if label.ndim > 1:
            label = label.squeeze()
        logits = self(
            normalized_total_p_RT=normalized_total_p_rt,
        )
        logits = logits.squeeze(1)
        if len(self.model_args.prediction_config.class_names) > 2:
            logits = logits.squeeze(-1).softmax(dim=1)
            loss = self.loss(logits, label)
        else:
            if len(logits.shape) > 1:
                logits = logits.squeeze()
            if len(logits.shape) == 0:
                logits = logits.unsqueeze(0)
            loss = self.loss(logits, label.float())
            logits = logits.sigmoid()
        return label, loss, logits
