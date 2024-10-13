"""TODO add docstring"""

import torch
from torch import nn

from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.model_specific_args.LitArgs import LitArgs
from src.configs.trainer_args import Base
from src.models.base_model import BaseModel


class LitModel(BaseModel):
    """
    TODO add docstring
    """

    def __init__(
        self,
        model_args: LitArgs,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(model_args, trainer_args)

        # Define the layers of the model
        embedding_dim = model_args.eyes_dim
        hidden_dim = model_args.model_params.hidden_dim
        dropout = model_args.model_params.dropout
        num_heads = model_args.model_params.num_heads
        encoder_ff_dim = model_args.model_params.encoder_ff_dim
        max_seq_len = model_args.max_seq_len

        self.fc0 = nn.Linear(embedding_dim, hidden_dim)
        intermediate_dim = hidden_dim // 2
        self.fc = nn.Linear(hidden_dim, intermediate_dim)
        output_dim = 1
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

        self.encoder = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            batch_first=True,
            dim_feedforward=encoder_ff_dim,
            dropout=dropout,
        )

        self.activation = nn.ReLU()

        # Define positional encoding
        self.pos_encoder = nn.Embedding(max_seq_len, embedding_dim)

        self.register_buffer(
            "cls_token", torch.randn(1, embedding_dim, requires_grad=True)
        )

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get device based on x
        # device = self.device
        # Add a cls token-like vector to the beginning of each sequence
        # x = [torch.cat((self.cls_token, seq), dim=0) for seq in x]

        # TODO add positional encoding
        # Add positional encoding to each vector in x based on its length
        # x = [seq + self.pos_encoder(torch.arange(len(seq), device=device)) for seq in x]

        # Define source key padding mask based on whether the value is 0
        src_key_padding_mask = x[:, :, 0] == 0

        # x = nn.utils.rnn.pad_sequence(x, batch_first=True)
        x = self.fc0(x)

        output = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Keep only the cls token-like vector (first vector in output)
        # output = output[:, 0, :]
        # Keep only the non-padding values from output
        # output = output[~src_key_padding_mask]

        # TODO currently the mean is taken over the entire sequence, including padding.
        # This is not ideal. One option is to use the mask to only take the mean over
        # non-padding values. Another option is to use the cls token-like vector.

        output = torch.mean(output, dim=1).squeeze(0)
        output = self.activation(output)
        output = self.activation(self.fc(output))
        output = self.fc2(output)
        return output

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)
        assert batch_data.eyes is not None, "eyes_tensor not in batch_dict"
        label = batch_data.labels

        if label.ndim > 1:
            label = label.squeeze()
        logits = self(x=batch_data.eyes)
        logits = logits.squeeze(1)
        loss = self.loss(logits, label.float())
        logits = logits.sigmoid()
        return label, loss, logits
