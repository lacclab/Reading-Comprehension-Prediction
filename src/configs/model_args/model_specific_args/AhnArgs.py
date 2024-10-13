from dataclasses import dataclass, field

from src.configs.constants import BackboneNames, ModelNames, ConfigName
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.prediction_modes import IsCorrectPredCfg, PredCfg
from src.configs.utils import register_config
from src.configs.constants import MAX_SCANPATH_LENGTH

GROUP = ConfigName.MODEL


@register_config(group=GROUP)
@dataclass
class AhnArgs(BaseModelArgs):
    batch_size: int = 100
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE  # TODO not in use
    use_fixation_report: bool = True
    text_dim: int = 0  # TODO not in use
    prediction_config: PredCfg = field(
        default_factory=lambda: IsCorrectPredCfg(use_eyes_only=True)
    )
    add_contrastive_loss: bool = False

    contrastive_loss_embd_dim: int = 0

    max_seq_len: int = 313
    max_eye_len: int = 313
    preorder: bool = False

    fixation_features = [
        "CURRENT_FIX_DURATION",
        "CURRENT_FIX_PUPIL",
        "CURRENT_FIX_X",
        "CURRENT_FIX_Y",
    ]


@dataclass
class AhnRNNParams(BaseModelParams):
    """
    Configuration for the Ahn RNN model.
    """

    model_name: ModelNames = ModelNames.AHN_RNN_MODEL
    hidden_dim: int = 512
    num_lstm_layers: int = 1
    fc_hidden_dim: int = 512
    fc_dropout: float = 0.3


@register_config(group=GROUP)
@dataclass
class AhnRNN(AhnArgs):
    model_params: AhnRNNParams = field(
        default_factory=lambda: AhnRNNParams(class_weights=None)
    )

    def __post_init__(self):
        super().__post_init__()
        self.contrastive_loss_embd_dim = self.model_params.hidden_dim


@dataclass
class AhnCNNParams(BaseModelParams):
    """
    Configuration for the Ahn CNN model.
    """

    model_name: ModelNames = ModelNames.AHN_CNN_MODEL
    hidden_dim: int = 40
    conv_kernel_size: int = 3
    pooling_kernel_size: int = 2
    fc_hidden_dim1: int = 50
    fc_hidden_dim2: int = 20
    fc_dropout: float = 0.3


@register_config(group=GROUP)
@dataclass
class AhnCNN(AhnArgs):
    model_params: AhnCNNParams = field(
        default_factory=lambda: AhnCNNParams(
            class_weights=None,
        )
    )

    def __post_init__(self):
        # do the post init of the parent class
        super().__post_init__()
        self.contrastive_loss_embd_dim = (
            (MAX_SCANPATH_LENGTH - 6) // 2
        ) * self.model_params.hidden_dim
