from dataclasses import dataclass, field

from src.configs.constants import ModelNames, ConfigName
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class MLPParams(BaseModelParams):
    """
    Configuration for the MLP model.

    Attributes
    ----------
    hidden_dim : int
        Dimension of the hidden layer in the model.
    lstm_dropout : float
        Dropout rate for the LSTM layers.
    fc_dropout : float
        Dropout rate for the fully connected layers.
    """

    model_name: ModelNames = ModelNames.MLP_MODEL
    hidden_dim: int = 128
    lstm_dropout: float = 0.1
    fc_dropout: float = 0.5


@register_config(group=GROUP)
@dataclass
class MLPArgs(BaseModelArgs):
    model_params: MLPParams = field(default_factory=MLPParams)
