"""MLPModel class (inherits from BaseModel)"""

import torch
from torch import nn

from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.model_specific_args.MLPArgs import MLPArgs
from src.configs.trainer_args import Base
from src.models.base_model import BaseModel


class MLPModel(BaseModel):
    """
    TODO add docstring
    """

    def __init__(
        self,
        model_args: MLPArgs,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(model_args, trainer_args)

        self.model_args = model_args
        self.input_size = model_args.eyes_dim
        if model_args.model_params.hidden_dim is None:
            self.hidden_size = model_args.eyes_dim
        else:
            self.hidden_size = model_args.model_params.hidden_dim
        self.output_size = 1
        self.max_len = model_args.max_seq_len * self.input_size

        # Define the layers
        self.fc1 = nn.Linear(self.max_len, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        highest_ind = max(b.nonzero(as_tuple=True)[0].max() for b in x)
        assert highest_ind <= self.max_len, "Highest index is greater than max_len"
        if self.model_args.prediction_config.use_eyes_only:
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), x.size(1), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = super().unpack_batch(batch)
        assert batch_data.eyes is not None, "eyes_tensor not in batch_dict"
        eyes = batch_data.eyes
        label = batch_data.labels

        if label.ndim > 1:
            label = label.squeeze()
        logits = self(x=eyes)
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
