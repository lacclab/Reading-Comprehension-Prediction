from dataclasses import dataclass, field

from src.configs.constants import ModelNames, ConfigName
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class LitParams(BaseModelParams):
    """
    Configuration for the Lit model.

    Attributes
    ----------
    num_heads : int
        Number of attention heads in the transformer model.
    hidden_dim : int
        Dimension of the hidden layer in the model.
    encoder_ff_dim : int
        Dimension of the feed-forward network in the transformer encoder.
    max_seq_len : int
        Maximum sequence length for the transformer model.
    dropout : float
        Dropout rate for regularization.
    """

    model_name: ModelNames = ModelNames.LIT_MODEL
    num_heads: int = 8
    hidden_dim: int = 128
    encoder_ff_dim: int = 128
    max_seq_len: int = 165  # Max number of words in a OneStopQA paragraph
    dropout: float = 0.25


@register_config(group=GROUP)
@dataclass
class LitArgs(BaseModelArgs):
    model_params: LitParams = field(default_factory=LitParams)
