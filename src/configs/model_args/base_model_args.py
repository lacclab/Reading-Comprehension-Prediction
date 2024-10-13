"""
This module contains dataclasses for defining model arguments and parameters.

The module defines a base configuration class `BaseModelParams` for common model parameters
and a base configuration class `BaseModelArgs` for model arguments shared by LitModel,
MLPModel, and others.

The configuration classes use constants from the `src.configs.constants` module, such as
`BackboneNames`, `DataRepresentation`, and `ModelNames`, as well as the `PredCfg` class
from the `src.configs.prediction_modes` module.

The `BaseModelArgs` class includes various attributes for configuring the model, such as
batch size, model parameters, backbone, input dimensions, prediction configuration,
sequence lengths, and feature lists for eye and fixation data.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING

from src.configs.constants import (
    BackboneNames,
    DataRepresentation,
    ItemLevelFeaturesModes,
    ModelNames,
    FeatureMode,
)
from src.configs.prediction_modes import (
    PredCfg,
)


@dataclass
class BaseModelParams:
    """
    Base configuration class for common model parameters.

    Attributes:
        model_name (ModelNames): The name of the model. Must be specified.
        class_weights (list[float] | None): Weights for each class in the loss function.
            None means equal weight for all classes. Default is None.
        concat_or_duplicate (DataRepresentation): The mode for handling the multiple choice selections.
            This can be either 'concat' to concatenate the data (e.g. {P||Q||A1||A2||A3||A4}),
            or 'duplicate' to duplicate the data (e.g. {P||Q||A1, P||Q||A2, P||Q||A3, P||Q||A4}).
            Default is DataRepresentation.CONCAT.
        prepend_eye_data (bool): A flag indicating whether to prepend the eye data to the input.
            If True, the eye data will be added at the beginning of the input. Default is False.
    """

    model_name: ModelNames = MISSING
    class_weights: list[float] | None = None
    concat_or_duplicate: DataRepresentation = DataRepresentation.CONCAT
    prepend_eye_data: bool = False
    item_level_features_mode: ItemLevelFeaturesModes = ItemLevelFeaturesModes.DAVID


@dataclass
class BaseModelArgs:
    """
    Base configuration class for model arguments shared by the models.

    Attributes:
        batch_size (int): The batch size for training and inference. Must be specified.
        model_params (BaseModelParams): The model parameters. Must be specified.
        backbone (BackboneNames): The backbone model to use. Must be specified.
        use_fixation_report (bool): A flag indicating whether to use the fixation report. Must be specified.
        text_dim (int): The dimension of the text input. Must be specified.
        prediction_config (PredCfg): The prediction configuration. Must be specified.
        max_seq_len (int): The maximum sequence length of the text input in tokens. Must be specified.
        max_eye_len (int): The maximum sequence length of the eye data in tokens. Must be specified.
        accumulate_grad_batches (int): The number of batches to accumulate gradients before updating the weights.
            Default is 1.
        eyes_dim (int): The dimension of the eye data. Initialized based on `ia_features`.
        fixation_dim (int): The dimension of the fixation data. Initialized based on `fixation_features` and `ia_features`.
        ia_features (list[str]): The list of word-level features to use.
            Default includes various word-level features.
        fixation_features (list[str]): The list of fixation features to use.
            Default includes various fixation features.
        ia_categorical_features (list[str]): The list of categorical word-level features.
            Default is an empty list.
        add_beyelstm_features (bool): A flag indicating whether to add BEyeLSTM features. Default is False.
        n_tokens (int): The number of tokens. Default is 0.
        eye_token_id (int): The ID of the eye token. Default is 0.
        sep_token_id (int): The ID of the separator token. Default is 0.
        is_training (bool): A flag indicating whether the model is in training mode. Default is False.
    """

    batch_size: int = MISSING
    model_params: BaseModelParams = MISSING
    backbone: BackboneNames = MISSING
    use_fixation_report: bool = MISSING
    text_dim: int = MISSING
    prediction_config: PredCfg = MISSING
    max_seq_len: int = MISSING  # longest text input sequence in the dataset in tokens
    max_eye_len: int = MISSING  # longest eye sequence in the dataset in tokens
    contrastive_loss_embd_dim: int = MISSING

    accumulate_grad_batches: int = 1
    eyes_dim: int = field(init=False)  # Defined according to ia_features
    fixation_dim: int = field(
        init=False
    )  # Defined according to fixation_features + ia_features
    feature_mode: FeatureMode = FeatureMode.EYES_WORDS
    word_features: list[str] = field(
        default_factory=lambda: [
            "gpt2_Surprisal",
            "Wordfreq_Frequency",
            "Length",
            "start_of_line",
            "end_of_line",
            "Is_Content_Word",
            "n_Lefts",
            "n_Rights",
            "Distance2Head",
        ]
    )
    eye_features: list[str] = field(
        default_factory=lambda: [
            "IA_DWELL_TIME",
            "IA_DWELL_TIME_%",
            "IA_FIXATION_%",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_RUN_COUNT",
            "IA_FIRST_FIXATION_DURATION",
            "IA_FIRST_FIXATION_VISITED_IA_COUNT",
            "IA_FIRST_RUN_DWELL_TIME",
            "IA_FIRST_RUN_FIXATION_COUNT",
            "IA_SKIP",
            "IA_REGRESSION_PATH_DURATION",
            "IA_REGRESSION_OUT_COUNT",
            "IA_SELECTIVE_REGRESSION_PATH_DURATION",
            "IA_LAST_FIXATION_DURATION",
            "IA_LAST_RUN_DWELL_TIME",
            "IA_LAST_RUN_FIXATION_COUNT",
            "IA_TOP",
            "IA_LEFT",
            "IA_FIRST_FIX_PROGRESSIVE",
            "normalized_ID",
            "PARAGRAPH_RT",
            "total_skip",
        ]
    )

    ia_features: list[str] = MISSING
    # "IA_ID", #! Doesn't work with fixation report (datamodule.py, line 627, in add_ia_report_features_to_fixation_data)
    # "IA_LABEL",
    # Features with NaNs: TODO handle and add?
    # IA_AVERAGE_FIX_PUPIL_SIZE 1080045
    # IA_FIRST_RUN_FIXATION_% 1080045
    # IA_FIRST_SACCADE_AMPLITUDE 1101865
    # IA_FIRST_SACCADE_ANGLE 1101865
    # IA_LAST_RUN_FIXATION_% 1080045
    # IA_LAST_SACCADE_AMPLITUDE 1095860
    # IA_LAST_SACCADE_ANGLE 1095860
    # Entity 2193970
    # prev_Wordfreq_Frequency 19438
    # prev_subtlex_Frequency 19438
    # prev_Length 19438
    # prev_gpt2_Surprisal 19438
    # regression_rate 1080045
    # normalized_part_dwell_time 8051
    # normalized_part_ID 397

    fixation_features: list[str] = field(
        default_factory=lambda: [
            "CURRENT_FIX_INDEX",
            "CURRENT_FIX_DURATION",
            "CURRENT_FIX_PUPIL",
            "CURRENT_FIX_X",
            "CURRENT_FIX_Y",
            "NEXT_FIX_ANGLE",
            "PREVIOUS_FIX_ANGLE",
            "NEXT_FIX_DISTANCE",
            "PREVIOUS_FIX_DISTANCE",
            "NEXT_SAC_AMPLITUDE",
            "NEXT_SAC_ANGLE",
            "NEXT_SAC_AVG_VELOCITY",
            "NEXT_SAC_DURATION",
            "NEXT_SAC_PEAK_VELOCITY",
            # "NEXT_SAC_BLINK_DURATION", # Mostly NaNs
        ]
    )
    # Categorical should be added in both places
    ia_categorical_features: list[str] = field(
        default_factory=lambda: [
            # "IA_ID",
            # "IA_LABEL",
        ]
    )
    add_beyelstm_features: bool = False
    n_tokens: int = 0
    eye_token_id: int = 0
    sep_token_id: int = 0
    is_training: bool = False

    #! TODO NOTE STILL MANY HARDCODED/AUTO-DERIVED VALUES IN THE CODE. CHECK BEFORE USING.
    preorder: bool = True  # Order the answers and convert labels according to ABCD order before model input

    def __post_init__(self):
        """
        Post-initialization hook to compute `eyes_dim` and `fixation_dim` based on the feature lists.

        Note: Changes made here are not applied if inheritance is used, unless `super().__post_init__()`
        is called in the derived class.
        # TODO if wandb changes a value that is used here (like backbone) then there can be a mismatch
        # TODO as __post_init__ is not run again. See current workaround in update_cfg_with_wandb.
        """
        if self.feature_mode == FeatureMode.EYES_WORDS:
            self.ia_features = self.eye_features + self.word_features
        elif self.feature_mode == FeatureMode.EYES:
            self.ia_features = self.eye_features
        elif self.feature_mode == FeatureMode.WORDS:
            self.ia_features = self.word_features

        n_categorical_features = len(self.ia_categorical_features)
        self.eyes_dim = len(self.ia_features) - n_categorical_features
        self.fixation_dim = len(self.fixation_features) + self.eyes_dim

        self.ia_features_to_add_to_fixation_data = self.ia_features

        self.text_dim: int = (
            768 if self.backbone == BackboneNames.ROBERTA_BASE else 1024
        )  # Updated in update_cfg_with_wandb as well

        # if contrastive_loss_embd_dim is still missing, set it to the text_dim
        if self.contrastive_loss_embd_dim == MISSING:
            self.contrastive_loss_embd_dim = self.text_dim
