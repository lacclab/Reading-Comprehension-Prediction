from dataclasses import dataclass, field

from src.configs.constants import BackboneNames, ModelNames, ConfigName
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.prediction_modes import ConditionPredCfg, PredCfg
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class FSEParams(BaseModelParams):
    """
    Configuration for the Fse model.

    Attributes
    ----------
    num_lstm_layers : int
        Number of LSTM layers in the model.
    lstm_dropout : float
        Dropout rate for the LSTM layers.
    fc_dropout : float
        Dropout rate for the fully connected layers.
    hidden_dim : int
        Dimension of the hidden layer in the model.
    """

    model_name: ModelNames = ModelNames.FSE_MODEL
    num_lstm_layers: int = 8
    lstm_dropout: float = 0.1
    fc_dropout: float = 0.5
    hidden_dim: int = 128


@register_config(group=GROUP)
@dataclass
class FSEArgs(BaseModelArgs):
    """
    Fixation Sequence Encoder Based Model class (inherits from BaseModel)
    """

    model_params: FSEParams = field(default_factory=FSEParams)
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    use_fixation_report: bool = True
    text_dim: int = 1024
    max_seq_len: int = 309
    max_eye_len: int = 309
