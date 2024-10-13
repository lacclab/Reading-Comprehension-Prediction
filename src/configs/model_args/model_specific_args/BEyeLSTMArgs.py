from dataclasses import dataclass, field

from src.configs.constants import BackboneNames, ConfigName, ModelNames
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.prediction_modes import (
    PredCfg,
    IsCorrectPredCfg,
)
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class BEyeLSTMParams(BaseModelParams):
    """
    Configuration for the BEye LSTM model.

    Attributes
    ----------
    num_pos (int): Number of positions.
    num_content (int): Number of content types.
    fixations_dim (int): Dimension of fixations.
    gsf_dim (int): Dimension of GSF.
    """

    model_name: ModelNames = ModelNames.BEYELSTM_MODEL
    num_pos: int = 5
    num_content: int = 2
    fixations_dim: int = 4  #! Not a hyperparameter to play with
    """
    Originally:
    **35** binned values X (**13** reading features + **5** linguistic features) + **4** global features = **634**
    Ours:
    **44** binned values X (**11** reading features + **5** linguistic features) + **4** global features = **708**
    """
    gsf_dim: int = 708
    dropout_rate: float = 0.3  # Dropout rate of fc1 and fc2
    embedding_dim: int = (
        8  # The embedding dimension for categorical features (POS, Content)
    )
    # The output dimensions for fc1,2 after each LSTM
    lstm_block_fc1_out_dim: int = 128  # originally: 50
    lstm_block_fc2_out_dim: int = 64  # originally: 20
    gsf_out_dim: int = 64  # orignally: 32
    # The middle embedding size of the FC after the concat of all LSTM results and gsf (all separate layers, only the dim is shared)
    after_cat_fc_hidden_dim: int = 64
    hidden_dim: int = 128  # the hidden dim inside the LSTM. Originally: 25


@register_config(group=GROUP)
@dataclass
class BEyeLSTMArgs(BaseModelArgs):
    """
    Model arguments for the BEyeLSTM model.
    """

    batch_size: int = 64
    backbone: BackboneNames = BackboneNames.ROBERTA_BASE  # TODO not in use
    use_fixation_report: bool = True
    text_dim: int = 0  # TODO not in use
    prediction_config: PredCfg = field(
        default_factory=lambda: IsCorrectPredCfg(use_eyes_only=True)
    )
    add_contrastive_loss: bool = False
    max_seq_len: int = 311  # in use only for creating TextDataset
    max_eye_len: int = 257  # TODO not in use
    model_params: BEyeLSTMParams = field(
        default_factory=lambda: BEyeLSTMParams(class_weights=None)
    )
    contrastive_loss_embd_dim: int = (
        BEyeLSTMParams.lstm_block_fc2_out_dim * 3 + BEyeLSTMParams.gsf_out_dim
    )

    add_beyelstm_features: bool = True
    fixation_features: list[str] = field(
        default_factory=lambda: [
            "CURRENT_FIX_DURATION",
            "CURRENT_FIX_PUPIL",
            "CURRENT_FIX_X",
            "CURRENT_FIX_Y",
            "NEXT_FIX_INTEREST_AREA_INDEX",
            "CURRENT_FIX_INTEREST_AREA_INDEX",
        ]
    )
    ia_features: list[str] = field(
        default_factory=lambda: [
            "Is_Content_Word",
            "Reduced_POS",
            "n_Lefts",
            "n_Rights",
            "Distance2Head",
            "Head_Direction",
            "gpt2_Surprisal",
            "Wordfreq_Frequency",
            "Length",
            "Entity",
            "POS",
            "TRIAL_IA_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
            "IA_FIRST_FIXATION_DURATION",
            "IA_DWELL_TIME",
        ]
    )

    ia_categorical_features: list[str] = field(
        default_factory=lambda: [
            "Is_Content_Word",
            "Reduced_POS",
            "Entity",
            "POS",
            "Head_Direction",
            "TRIAL_IA_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
        ]
    )


# @register_config(group=GROUP)
# @dataclass
# class BEyeLSTMQCond(BEyeLSTMArgs):
#     prediction_config: PredCfg = field(
#         default_factory=lambda: QConditionPredCfg(use_eyes_only=True)
#     )
#     batch_size: int = 256
#     accumulate_grad_batches: int = 1


# @register_config(group=GROUP)
# @dataclass
# class BEyeLSTMIsCorrectCLSampling(BEyeLSTMArgs):
#     add_contrastive_loss: bool = True
#     prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)

# @register_config(group=GROUP)
# @dataclass
# class BEyeLSTM(BEyeLSTMArgs):
#     # accumulate_grad_batches: int = 1
#     add_contrastive_loss: bool = False
#     prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
#     # max_seq_len: int = 313
