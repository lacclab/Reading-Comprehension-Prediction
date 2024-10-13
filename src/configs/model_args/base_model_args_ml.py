from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from src.configs.constants import (
    MLModelNames,
    ItemLevelFeaturesModes,
)
from src.configs.prediction_modes import (
    PredCfg,
)


@dataclass
class BaseMLModelParams:
    """
    Base configuration for all machine learning models.

    Attributes
    ----------
    class_weights : list[float] | None
        Weights for each class in the loss function. None means equal weight for all classes.
    """

    model_name: MLModelNames = MISSING
    class_weights: list[float] | None = None
    sklearn_pipeline: Any = MISSING
    sklearn_pipeline_params: dict = MISSING
    prepend_eye_data: bool = False
    concat_or_duplicate: str = MISSING

    sklearn_pipeline_params = field(default_factory=dict)

    item_level_features_mode: ItemLevelFeaturesModes = ItemLevelFeaturesModes.NONE
    # for each additional_feature, add a line here
    # with boolean value to indicate if the feature is to be used
    use_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT: bool = False
    use_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT: bool = False

    #####################################################################
    def init_sklearn_pipeline_params(self):
        # create sklearn pipeline params
        # pass over the attributes of the class and add them to the pipeline params
        for key, value in self.__dict__.items():
            if key.startswith("sklearn_pipeline_param_"):
                self.sklearn_pipeline_params[
                    key.replace("sklearn_pipeline_param_", "")
                ] = value

        # setting the params is done in the model.__init__


@dataclass
class BaseMLModelArgs:
    """
    Basic model arguments for the machine learning models.
    """

    batch_size: int = MISSING
    model_params: BaseMLModelParams = MISSING
    prediction_config: PredCfg = MISSING
    text_dim: int = MISSING
    max_seq_len: int = MISSING  # longest text input sequence in the dataset in tokens
    max_eye_len: int = MISSING  # longest eye sequence in the dataset in tokens
    backbone: str = MISSING

    # TODO Add more model-specific arguments here
    # TODO .cont. and data specific arguments
    use_fixation_report: bool = MISSING

    eyes_dim: int = field(init=False)  # Defined according to ia_features
    fixation_dim: int = field(
        init=False
    )  # Defined according to fixation_features + ia_features
    ia_features: list[str] = field(
        default_factory=lambda: [
            "IA_FIRST_FIXATION_DURATION",
            "TRIAL_IA_COUNT",
            "IA_FIXATION_%",
            "PARAGRAPH_RT",
            "IA_SKIP",
            "start_of_line",
            "IA_FIRST_RUN_DWELL_TIME",
            "Is_Content_Word",
            "IA_TOP",
            "n_Lefts",
            "Entity",
            "IA_RUN_COUNT",
            "IA_FIRST_FIX_PROGRESSIVE",
            "IA_FIRST_FIXATION_VISITED_IA_COUNT",
            "Reduced_POS",
            "gpt2_Surprisal",
            "IA_REGRESSION_OUT_COUNT",
            "IA_DWELL_TIME",
            "IA_LAST_FIXATION_DURATION",
            "IA_LAST_RUN_FIXATION_COUNT",
            "IA_SELECTIVE_REGRESSION_PATH_DURATION",
            "normalized_ID",
            "POS",
            "Head_Direction",
            "total_skip",
            "n_Rights",
            "IA_REGRESSION_PATH_DURATION",
            "IA_LEFT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
            "Distance2Head",
            "Length",
            "end_of_line",
            "IA_FIRST_RUN_FIXATION_COUNT",
            "IA_LAST_RUN_DWELL_TIME",
            "Wordfreq_Frequency",
            "IA_DWELL_TIME_%",
        ]
    )
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
            "Reduced_POS",
            "NEXT_FIX_INTEREST_AREA_INDEX",
            "CURRENT_FIX_INTEREST_AREA_INDEX",
            # "NEXT_SAC_BLINK_DURATION", # Mostly NaNs
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
            "TRIAL_IA_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
        ]
    )

    add_beyelstm_features: bool = True
    n_tokens: int = 0
    eye_token_id: int = 0
    sep_token_id: int = 0
    is_training: bool = False

    def __post_init__(self):
        # Note changes here are not applied if inheritance is used! unless you call super().__post_init__()
        n_categorial_features = len(self.ia_categorical_features)
        self.eyes_dim = len(self.ia_features) - n_categorial_features
        self.fixation_dim = len(self.fixation_features) + self.eyes_dim

        self.ia_features_to_add_to_fixation_data = self.ia_features
