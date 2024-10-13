from dataclasses import dataclass, field

from src.configs.constants import BackboneNames, ModelNames, ConfigName
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.prediction_modes import ChosenAnswerPredCfg, PredCfg, IsCorrectPredCfg
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class EyettentionParams(BaseModelParams):
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

    model_name: ModelNames = ModelNames.EYETTENTION_MODEL
    LSTM_hidden_dim: int = 128
    num_LSTM_layers: int = 8
    LSTM_dropout: float = 0.2
    fc_dropout: float = 0.2
    # fc_dims: list[int] = field(
    #     default_factory=lambda: [512, 256, 256, 256]
    # )  # ? Not in use
    embedding_dropout: float = 0.2


@register_config(group=GROUP)
@dataclass
class EyettentionArgs(BaseModelArgs):
    """
    Model arguments for the Eyettention model.
    """

    batch_size: int = 64
    accumulate_grad_batches: int = 4
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
    use_fixation_report: bool = True
    text_dim: int = 768 if backbone == BackboneNames.ROBERTA_BASE else 1024
    max_seq_len: int = 300
    max_eye_len: int = 300
    add_contrastive_loss: bool = False

    model_params: EyettentionParams = field(
        default_factory=lambda: EyettentionParams(
            model_name=ModelNames.EYETTENTION_MODEL,
            class_weights=None,  # Gets overritten by the trainer
        )
    )

    fixation_features: list[str] = field(
        default_factory=lambda: [  #! Keep the order of the first 3 here - the model assumes their existance in this order
            "CURRENT_FIX_INTEREST_AREA_INDEX",
            "CURRENT_FIX_DURATION",
            "CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE",
        ]
    )
    ia_features: list[str] = field(
        default_factory=lambda: [
            "IA_FIRST_FIXATION_DURATION",
            "IA_DWELL_TIME",
        ]
    )

    ia_categorical_features: list[str] = field(  # They are exluded
        default_factory=lambda: [
            "Is_Content_Word",
            "Reduced_POS",
            "Entity",
            "POS",
            "Head_Direction",
            "TRIAL_IA_COUNT",
            "TRIAL_IA_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
        ]
    )


@register_config(group=GROUP)
@dataclass
class Eyettention(EyettentionArgs):
    add_contrastive_loss: bool = False
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
    max_seq_len: int = 313
