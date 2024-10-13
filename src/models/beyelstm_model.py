"""Beye LSTM baseline model.
Based on https://github.com/aeye-lab/etra-reading-comprehension/blob/main/nn/model.py
"""

from typing import Tuple

from src.models.base_model import BaseModel
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_metric_learning import losses

from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.constants import PredMode
from src.configs.model_args.model_specific_args.BEyeLSTMArgs import (
    BEyeLSTMArgs,
    BEyeLSTMParams,
)
from src.configs.trainer_args import Base
from src.configs.constants import MAX_SCANPATH_LENGTH


class LSTMBlock(nn.Module):
    """LSTM block for the Beye model."""

    def __init__(
        self,
        model_params: BEyeLSTMParams,
        input_dim: int | None = None,
        num_embed: int | None = None,
    ) -> None:
        """Initialize LSTMBlock.

        Args:
            model_params (ModelParams): Model parameters.
            input_dim (int | None, optional): Input dimension. Defaults to None.
            num_embed (int | None, optional): Embedding dimension. Defaults to None.
        """
        super().__init__()
        assert (input_dim is None) != (
            num_embed is None
        ), "input_dim and num_embeddings cannot both be None or not None."
        self.num_embeddings = num_embed  # for POS and Content
        if num_embed:
            self.embedding = nn.Embedding(num_embed, model_params.embedding_dim)
            lstm_input_dim = model_params.embedding_dim
        else:  # for Fixations
            lstm_input_dim = input_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            model_params.hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(model_params.dropout_rate)
        self.fc1 = nn.Linear(
            2 * model_params.hidden_dim, model_params.lstm_block_fc1_out_dim
        )
        self.fc2 = nn.Linear(
            model_params.lstm_block_fc1_out_dim, model_params.lstm_block_fc2_out_dim
        )
        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor, seq_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for LSTMBlock.

        Args:
            seq_lengths (torch.Tensor | None): Length of scanpath for each trial. Defaults to None.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.num_embeddings:
            x = self.embedding(x)

        if seq_lengths is not None:
            sorted_lengths, indices = torch.sort(seq_lengths, descending=True)
            x = x[indices]
            # Pass the entire sequence through the LSTM layer
            packed_x = nn.utils.rnn.pack_padded_sequence(
                input=x,
                lengths=sorted_lengths.to("cpu"),
                batch_first=True,
                enforce_sorted=True,
            )
            assert not torch.isnan(packed_x.data).any()

            packed_output, (ht, ct) = self.lstm(packed_x)

            # from dimension (2, batch_size, hidden_dim) to (batch_size, 2*hidden_dim)
            x = torch.cat((ht[0], ht[1]), dim=1)
            x = x[torch.argsort(indices)]
        else:
            output, (h, c) = self.lstm(x)
            h_concat = torch.cat((h[0], h[1]), dim=1)
            x = h_concat

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x


class BEyeLSTMModel(BaseModel):
    """Beye model."""

    # TODO move to top of file

    def __init__(
        self,
        model_args: BEyeLSTMArgs,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ) -> None:
        super().__init__(model_args, trainer_args)
        self.preorder = False
        self.model_args = model_args
        model_params = model_args.model_params
        self.pos_block = LSTMBlock(model_params, num_embed=model_params.num_pos)
        self.content_block = LSTMBlock(model_params, num_embed=model_params.num_content)
        self.fixations_block = LSTMBlock(
            model_params, input_dim=model_params.fixations_dim
        )

        self.gsf_block = nn.Sequential(
            nn.Dropout(p=model_params.dropout_rate),
            nn.Linear(
                in_features=model_params.gsf_dim, out_features=model_params.gsf_out_dim
            ),
            nn.ReLU(),
        )
        fc1_in_features = (
            model_params.lstm_block_fc2_out_dim * 3 + model_params.gsf_out_dim
        )
        self.fc1 = nn.Linear(
            in_features=fc1_in_features,
            out_features=model_params.after_cat_fc_hidden_dim,
        )
        self.fc2 = nn.Linear(
            in_features=model_params.after_cat_fc_hidden_dim,
            out_features=self.num_classes,
        )

        if self.class_weights is not None:
            # For binary classification, we can use the standard cross-entropy loss,
            # otherwise we need to use the custom weighted loss
            # loss = self.calculate_weighted_loss(
            #     logits=logits, labels=labels, ordered_labels=labels
            # )
            self.loss = nn.CrossEntropyLoss(weight=self.class_weights)

        print(f"##### Preorder labels: {self.preorder} #####")
        if self.model_args.add_contrastive_loss:
            self.cl_alpha = 1
            self.cl_memory_size = 1024
            self.cl_temperature = 0.07
            self.contrastive_loss = losses.CrossBatchMemory(
                loss=losses.NTXentLoss(temperature=self.cl_temperature),
                embedding_size=model_params.lstm_block_fc2_out_dim * 3
                + model_params.gsf_out_dim,
                memory_size=self.cl_memory_size,
            )
            print(f"##### Contrastive loss added with alpha: {self.cl_alpha} #####")
            print(f"##### Contrastive loss memory size: {self.cl_memory_size} #####")
            print(f"##### Contrastive loss temperature: {self.cl_temperature} #####")
            print(
                f"##### Contrastive loss embedding size: {model_args.contrastive_loss_embd_dim} #####"
            )

        self.save_hyperparameters()

    def forward(  # type: ignore
        self,
        x_pos: torch.Tensor,
        x_content: torch.Tensor,
        x_gsf: torch.Tensor,
        x_fixations: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for NNModel.

        Args:
            x_pos (torch.Tensor): Position tensor.
            x_content (torch.Tensor): Content tensor.
            x_gsf (torch.Tensor): GSF tensor.
            x_fixations (torch.Tensor): Fixations tensor (batch size x MAX_SCANPATH_LEN x 4).
                                        Padded with 0s
            seq_lengths (torch.Tensor): Length of scanpath for each trial.

        Returns:
            torch.Tensor: Output tensor.
        """
        concat_list = []
        concat_list.append(self.pos_block(x_pos, seq_lengths=seq_lengths))
        concat_list.append(self.content_block(x_content, seq_lengths=seq_lengths))
        concat_list.append(self.gsf_block(x_gsf.squeeze()))
        concat_list.append(self.fixations_block(x_fixations, seq_lengths=seq_lengths))
        trial_embd = torch.cat(concat_list, dim=1)
        x = F.relu(self.fc1(trial_embd))
        x = self.fc2(x)
        return x, trial_embd

    def shared_step(
        self,
        batch: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)
        assert batch_data.fixation_features is not None, "eyes_tensor not in batch_dict"
        assert batch_data.scanpath_pads is not None, "scanpath_pads not in batch_dict"
        labels = batch_data.labels

        shortest_scanpath_pad = batch_data.scanpath_pads.min()
        longest_batch_scanpath: int = int(MAX_SCANPATH_LENGTH - shortest_scanpath_pad)

        fixation_features = batch_data.fixation_features[
            ..., :longest_batch_scanpath, :
        ]
        # scanpath_lengths = (
        #     batch_data.fixation_features.shape[1] - batch_data.scanpath_pads
        # )
        logits, trial_embd = self(
            x_fixations=fixation_features[..., :4],
            x_content=fixation_features[..., -2].int(),
            x_pos=fixation_features[..., -1].int(),
            x_gsf=batch_data.trial_level_features,
            seq_lengths=None,  # can use scanpath_lengths
            # seq_lengths=scanpath_lengths, # TODO - check if this is correct
        )

        if (
            self.prediction_mode in (PredMode.CHOSEN_ANSWER, PredMode.QUESTION_LABEL)
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
            loss = self.loss(logits.squeeze(), labels)

        if self.model_args.add_contrastive_loss:
            if trial_embd.ndim == 1:
                trial_embd = trial_embd.unsqueeze(0)
            contrastive_loss = self.contrastive_loss(trial_embd, ordered_labels)
            loss += self.cl_alpha * contrastive_loss
        return ordered_labels, loss, ordered_logits.squeeze(), labels, logits.squeeze()
