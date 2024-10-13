"""Ahn et al. baseline models
Based on
https://github.com/aeye-lab/etra-reading-comprehension/blob/main/ahn_baseline/evaluate_ahn_baseline.py
https://github.com/ahnchive/SB-SAT/blob/master/model/model_training.ipynb
"""

import torch
from pytorch_metric_learning import losses
from torch import nn

from src.configs.constants import MAX_SCANPATH_LENGTH, PredMode
from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.model_specific_args.AhnArgs import AhnRNN, AhnArgs, AhnCNN
from src.configs.trainer_args import Base
from src.models.base_model import BaseModel


class AhnModel(BaseModel):
    """
    Base model for Ahn et al.
    """

    def __init__(
        self,
        model_args: AhnArgs,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(model_args, trainer_args)
        self.model_args = model_args
        self.input_dim = (
            model_args.fixation_dim
            if model_args.use_fixation_report
            else model_args.eyes_dim
        )
        self.preorder = model_args.preorder
        self.model: nn.Module
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)

        if self.model_args.add_contrastive_loss:
            self.cl_alpha = 1
            self.cl_memory_size = 1024
            self.cl_temperature = 0.07
            self.contrastive_loss = losses.CrossBatchMemory(
                loss=losses.NTXentLoss(temperature=self.cl_temperature),
                embedding_size=model_args.contrastive_loss_embd_dim,
                memory_size=self.cl_memory_size,
            )
            print(f"##### Contrastive loss added with alpha: {self.cl_alpha} #####")
            print(f"##### Contrastive loss memory size: {self.cl_memory_size} #####")
            print(f"##### Contrastive loss temperature: {self.cl_temperature} #####")
            print(f"##### Contrastive loss embedding size: {model_args.text_dim} #####")

        self.save_hyperparameters()

    def forward(self, x):
        raise NotImplementedError

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)
        assert batch_data.fixation_features is not None, "eyes_tensor not in batch_dict"
        labels = batch_data.labels
        logits, hidden_representations = self(x=batch_data.fixation_features)

        if (
            self.prediction_mode
            in (
                PredMode.CHOSEN_ANSWER,
                PredMode.QUESTION_LABEL,
                PredMode.QUESTION_n_CONDITION,
            )
            and not self.preorder
        ):
            assert batch_data.answer_mappings is not None
            ordered_labels, ordered_logits = self.order_labels_logits(
                logits=logits, labels=labels, answer_mapping=batch_data.answer_mappings
            )
        else:
            ordered_labels = labels
            ordered_logits = logits

        if self.class_weights is not None:
            loss = self.calculate_weighted_loss(
                logits=logits, labels=labels, ordered_labels=ordered_labels
            )
        else:
            loss = self.loss(logits, labels)

        if self.model_args.add_contrastive_loss:
            last_hidden = hidden_representations
            contrastive_loss = self.contrastive_loss(last_hidden, ordered_labels)
            loss += self.cl_alpha * contrastive_loss

        return ordered_labels, loss, ordered_logits, labels, logits


class AhnRNNModel(AhnModel):
    """
    RNN model for Ahn et al. baseline
    """

    def __init__(
        self,
        model_args: AhnRNN,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(
            model_args, trainer_args, data_args=data_args, data_path_args=data_path_args
        )

        self.model_params = model_args.model_params
        self.hidden_size = 25
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=self.model_params.num_lstm_layers,
        )
        self.fc = nn.Sequential(
            nn.Dropout(self.model_params.fc_dropout),  # (batch_size, hidden_size * 2)
            nn.Linear(
                self.hidden_size * 2, self.model_params.fc_hidden_dim
            ),  # (batch_size, 50)
            nn.ReLU(),
            nn.Dropout(self.model_params.fc_dropout),
            nn.Linear(
                self.model_params.fc_hidden_dim, self.num_classes
            ),  # (batch_size, 2)
            nn.ReLU(),
        )

    def forward(self, x):
        # take the last hidden state of the lstm
        x, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        x = x[:, -1, :]  # (batch_size, hidden_size * 2)
        hidden_representations = x.clone()
        x = self.fc(x)  # (batch_size, 2)
        return x, hidden_representations


class AhnCNNModel(AhnModel):
    """
    CNN model for Ahn et al. baseline
    """

    def __init__(
        self,
        model_args: AhnCNN,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(
            model_args, trainer_args, data_args=data_args, data_path_args=data_path_args
        )

        self.input_dim = self.input_dim
        self.model_params = model_args.model_params
        hidden_dim = self.model_params.hidden_dim
        kernel_size = self.model_params.conv_kernel_size
        fc_dropout = self.model_params.fc_dropout
        fc_hidden_dim1 = self.model_params.fc_hidden_dim1
        fc_hidden_dim2 = self.model_params.fc_hidden_dim2

        self.conv_model = nn.Sequential(
            # (batch size, number of features, max seq len)
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
            ),  # (batch size, hidden_dim, max seq len - 2)
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size
            ),  # (batch size, hidden_dim, max seq len - 4)
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size
            ),  # (batch size, hidden_dim, max seq len - 6)
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=self.model_params.pooling_kernel_size
            ),  # (batch size, hidden_dim, (max seq len -6) / 2)
            nn.Dropout(fc_dropout),  # (batch size, hidden_dim, (max seq len -6) / 2)
            nn.Flatten(),  # (batch size, hidden_dim * ((max seq len -6) / 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(
                ((MAX_SCANPATH_LENGTH - 6) // 2) * hidden_dim, fc_hidden_dim1
            ),  # (batch size, 50)
            nn.ReLU(),
            nn.Dropout(fc_dropout),  # (batch size, 50)
            nn.Linear(fc_hidden_dim1, fc_hidden_dim2),  # (batch size, 20)
            nn.ReLU(),
            nn.Linear(fc_hidden_dim2, self.num_classes),  # (batch size, 2)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch size, number of features, max seq len)
        hidden_representations = self.conv_model(x)
        x = self.fc(hidden_representations)
        return x, hidden_representations
