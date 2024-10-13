"""Fixation Sequence Encoder Based Model class (inherits from BaseModel)"""

import warnings

import torch
from torch import nn
from transformers import RobertaModel

from src.configs.constants import MAX_SCANPATH_LENGTH
from src.configs.model_args.model_specific_args.FSEArgs import FSEArgs
from src.configs.trainer_args import Base
from src.models.base_model import BaseModel
from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs


class FixationSequenceEncoder(nn.Module):
    """
    TODO add docstring
    """

    def __init__(
        self,
        text_dim: int,
        f_input_dim: int,
        hidden_lstm_dim: int,
        max_len: int,
        num_lstm_layers: int,
        lstm_dropout: float,
    ):
        super().__init__()
        self.max_len = max_len
        # Create a 8-layer LSTM with dropout between the first 7 layers
        self.lstm = nn.LSTM(
            input_size=text_dim + f_input_dim,
            hidden_size=hidden_lstm_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        # initiate a learned positional embedding for all positions between 0 and max_len
        self.positional_encoder = nn.Embedding(max_len + 1, text_dim, padding_idx=0)

    def forward(
        self,
        fixation_features: torch.Tensor,
        scanpath_word_embds: torch.Tensor,
        scanpaths: torch.Tensor,  # pylint: disable=unused-argument
        scanpath_pads: torch.Tensor,
    ) -> torch.Tensor:
        """Padding idx for nn.embedding can't be negative (the padding value for the scanpath is -1)
        so we set padding idx=0 and add 1 to all scanpath values"""
        # positional_embds = self.positional_encoder(scanpaths + 1)
        # scanpath_word_embds += positional_embds

        # concat the fixation features with the scanpath word embeddings
        fixation_features = torch.cat(
            (fixation_features, scanpath_word_embds), dim=-1
        )  # (batch_size, max_len, text_dim + f_input_dim)

        # Pass the fixation features through the LSTM (take the pads into account)
        scanpath_lengths = (
            torch.ones(
                fixation_features.shape[0],
                dtype=torch.long,
                device=scanpath_pads.device,
            )
            * self.max_len
        )
        scanpath_lengths = scanpath_lengths - scanpath_pads

        fixation_features = nn.utils.rnn.pack_padded_sequence(
            input=fixation_features,
            lengths=scanpath_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )  # type: ignore

        assert not torch.isnan(fixation_features[0]).any()
        unused_output, (hidden, unused_cell) = self.lstm(fixation_features)  # pylint: disable=not-callable

        #! Notice that if I use the pack_padded_sequence, the row below is needed (I think)
        # padded_output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,
        #                                                   total_length=self.max_len)[0]

        # Keep the batch dimension and average over second dimension (the scanpath dimension)
        # ? hidden is (num_layers, batch_size, hidden_size), average over num_layers?????
        hidden = hidden.mean(dim=0)

        return hidden


class FSEModel(BaseModel):
    """
    TODO add docstring and move to  top of file
    """

    def __init__(
        self,
        model_args: FSEArgs,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(model_args, trainer_args)

        self.model_args = model_args
        self.input_size = model_args.eyes_dim
        self.hidden_size = model_args.model_params.hidden_dim
        class_names = model_args.prediction_config.class_names
        self.num_classes = len(class_names)
        self.output_size = self.num_classes
        self.max_seq_len = model_args.max_seq_len
        self.max_len = self.max_seq_len * self.input_size

        self.static_roberta = RobertaModel.from_pretrained(model_args.backbone)
        for param in self.static_roberta.parameters():
            param.requires_grad = False

        self.fse = FixationSequenceEncoder(
            text_dim=model_args.text_dim,
            f_input_dim=model_args.fixation_dim,
            hidden_lstm_dim=self.hidden_size,
            max_len=MAX_SCANPATH_LENGTH,
            num_lstm_layers=model_args.model_params.num_lstm_layers,
            lstm_dropout=model_args.model_params.lstm_dropout,
        )

        fc_dropout = model_args.model_params.fc_dropout
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.BatchNorm1d(self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(self.hidden_size // 4, self.output_size),
        )
        self.save_hyperparameters()

    def forward(
        self,
        fixation_features: torch.Tensor,
        scanpath_word_embds: torch.Tensor,
        scanpaths: torch.Tensor,
        scanpath_pads: torch.Tensor,
    ) -> torch.Tensor:
        encoded_sequences = self.fse(
            fixation_features, scanpath_word_embds, scanpaths, scanpath_pads
        )

        x = self.fc(encoded_sequences)
        return x

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)

        f_features = batch_data.fixation_features
        scanpaths = batch_data.scanpath
        scanpaths_pads = batch_data.scanpath_pads
        p_input_ids = batch_data.paragraph_input_ids
        p_attention_mask = batch_data.paragraph_input_masks
        inversions = batch_data.inversions
        inversions_pads = batch_data.inversions_pads
        label = batch_data.labels

        assert p_input_ids is not None, "p_input_ids is None"
        assert p_attention_mask is not None, "p_attention_mask is None"
        assert inversions is not None, "inversions is None"
        assert inversions_pads is not None, "inversions_pads is None"
        assert label is not None, "label is None"
        assert scanpaths is not None, "scanpaths is None"
        assert scanpaths_pads is not None, "scanpaths_pads is None"

        # TODO Move to preprocessing?
        with torch.no_grad():
            last_hidden_state_p_embds = self.static_roberta(
                p_input_ids, p_attention_mask
            )["last_hidden_state"]
            cls_embd = last_hidden_state_p_embds[:, 0, :]
            p_embds = last_hidden_state_p_embds[:, 1:, :]

            scanpath_word_embds = self.extract_scanpath_word_embds(
                scanpaths, p_embds, scanpaths_pads, inversions, inversions_pads
            )

            # Add The CLS vector as a "first fixation" which it's fixation features vector
            # is the mean of all fixation features
            scanpath_word_embds_with_cls = torch.cat(
                (cls_embd.unsqueeze(1), scanpath_word_embds), dim=1
            )
            f_features_with_cls = torch.cat(
                (torch.mean(f_features, dim=1).unsqueeze(1), f_features), dim=1
            )
            scanpaths_with_cls = torch.cat(
                (
                    torch.zeros(
                        scanpaths.shape[0], 1, dtype=torch.long, device=self.device
                    ),
                    scanpaths,
                ),
                dim=1,
            )
            scanpaths_pads_with_cls = scanpaths_pads + 1

        # Inversions: For each paragraph a list, where the value at entry i is the
        # IA_ID that is associated with the i-thtoken in the paragraph (batch_size, MAX_SEQ_LENGTH)

        logits = self(
            fixation_features=f_features_with_cls,
            scanpath_word_embds=scanpath_word_embds_with_cls,
            scanpaths=scanpaths_with_cls,
            scanpath_pads=scanpaths_pads_with_cls,
        )

        loss = self.loss(logits, label)

        return label, loss, logits

    def extract_scanpath_word_embds(
        self,
        scanpaths: torch.Tensor,
        p_embds: torch.Tensor,
        scanpath_pads: torch.Tensor,
        inversions: torch.Tensor,
        inversions_pads: torch.Tensor,
    ) -> torch.Tensor:
        """Extracts the word embeddings of the scanpath tokens from the paragraph embeddings.
        scanpath - contains IA_IDs (in the order of the fixation scanpath)
        p_embds - embeddings for each token in the paragraph
        Args:
            scanpath (torch.tensor): The scanpath tokens.
            p_embds (torch.tensor): The paragraph embeddings.
            inversions (torch.tensor): The inversions.
        Returns:
            torch.tensor: The scanpath word embeddings.
        """
        embds_list = []
        for scanpath, inversion, inversion_pad, p_embd, scanpath_pad in zip(
            scanpaths, inversions, inversions_pads, p_embds, scanpath_pads
        ):
            ia_embds = []
            for j in range(torch.max(scanpath).item() + 1):  # type: ignore
                indices = torch.where(
                    inversion[: self.max_seq_len - inversion_pad] == j
                )[0]  # The token indices that IA j is associated with
                ia_embds.append(torch.mean(p_embd[indices, :], dim=0))

            embds = torch.stack(ia_embds)[
                scanpath[: MAX_SCANPATH_LENGTH - scanpath_pad]
            ]

            pads_tensor = torch.zeros(
                scanpath_pad,  # type: ignore
                p_embd.shape[-1],
                device=self.device,
            )
            embds = torch.cat((embds, pads_tensor), dim=0)

            if torch.isnan(embds).any():
                warnings.warn(
                    """A NaN in text embds for each fixation!!! 
                    There is some IA_ID that is in the scanpath that is not in
                inversions. This means that no token is associated with this IA_ID. 
                This is not supposed to happen. 
                Covering by adding the average of all tokens in the paragraph instead of nan embds.
                """
                )
                evg_embd_vec = torch.mean(
                    p_embd[: self.max_seq_len - inversion_pad, :], dim=0
                )
                nan_indices = torch.where(torch.isnan(embds).any(dim=1))[0]
                for nan_idx in nan_indices:
                    embds[nan_idx] = evg_embd_vec

            assert not torch.isnan(embds).any()
            embds_list.append(embds)
        all_embds = torch.stack(embds_list)

        return all_embds
