from dataclasses import dataclass, field

from src.configs.constants import (
    BackboneNames,
    DataRepresentation,
    ModelNames,
    ConfigName,
)
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.prediction_modes import (
    ChosenAnswerPredCfg,
    IsCorrectPredCfg,
    PredCfg,
)  # , ChosenAnswerPredCfg
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class PostFusionParams(BaseModelParams):
    """
    Configuration for the PostFusion model.

    Attributes
    ----------
    concat_or_duplicate : MagModelModes
        The mode for handling the multiple-choice data.
        This can be either 'concat' to concatenate the data, or 'duplicate' to duplicate the data.
    prepend_eye_data : bool
        A flag indicating whether to prepend the eye data to the input.
        If True, the eye data will be added at the beginning of the input (otherwise no eyes!)
    """

    model_name: ModelNames = ModelNames.POSTFUSION_MODEL
    concat_or_duplicate: DataRepresentation = DataRepresentation.CONCAT
    prepend_eye_data: bool = False
    eye_projection_dropout: float = 0.1
    cross_attention_dropout: float = 0.1
    use_attn_mask: bool = True


# @dataclass
# class PostFusionArgs(BaseModelArgs):
#     """PostFusion Model"""

#     model_params: PostFusionParams = field(
#         default_factory=lambda: PostFusionParams(
#             class_weights=[1.0],  # Gets overwritten by trainer
#         )
#     )
#     max_seq_len: int = 300
#     prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
#     max_eye_len: int = 300  # Not in use in the model but still necessary. Don't take it so seriously though :^)
#     backbone: BackboneNames = BackboneNames.ROBERTA_RACE
#     use_fixation_report: bool = True
#     batch_size: int = 4
#     accumulate_grad_batches: int = 4
#     add_contrastive_loss: bool = False

#     sep_token_id: int = 2
#     is_training: bool = False
#     freeze: bool = False


# @register_config(group=GROUP)
# @dataclass
# class PostFusionCLArgs(BaseModelArgs):
#     """PostFusion Model"""

#     model_params: PostFusionParams = field(
#         default_factory=lambda: PostFusionParams(
#             class_weights=None,
#         )
#     )
#     add_contrastive_loss: bool = True


# @register_config(group=GROUP)
# @dataclass
# class PostFusionReadingComp(BaseModelArgs):
#     """PostFusion Model"""

#     model_params: PostFusionParams = field(
#         default_factory=lambda: PostFusionParams(
#             class_weights=None,
#         )
#     )
#     add_contrastive_loss: bool = False


@register_config(group=GROUP)
@dataclass
class PostFusion(BaseModelArgs):
    """PostFusion Model"""

    model_params: PostFusionParams = field(
        default_factory=lambda: PostFusionParams(
            class_weights=None,
        )
    )
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
    max_seq_len: int = 313
    max_eye_len: int = 313
    batch_size: int = 4
    accumulate_grad_batches: int = 16 // batch_size
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    use_fixation_report: bool = True
    add_contrastive_loss: bool = False
    sep_token_id: int = 2
    is_training: bool = False
    freeze: bool = False


@register_config(group=GROUP)
@dataclass
class PostFusionAnswers(PostFusion):
    prediction_config: PredCfg = field(
        default_factory=lambda: IsCorrectPredCfg(add_answers=True)
    )


@register_config(group=GROUP)
@dataclass
class PostFusionMultiClass(PostFusion):
    prediction_config: PredCfg = field(
        default_factory=lambda: ChosenAnswerPredCfg(add_answers=False)
    )


@register_config(group=GROUP)
@dataclass
class PostFusionAnswersMultiClass(PostFusion):
    prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)


@register_config(group=GROUP)
@dataclass
class PostFusionNoLinguistic(PostFusion):
    ia_features: list[str] = field(
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
            "IA_FIRST_FIX_PROGRESSIVE",
            "PARAGRAPH_RT",
            "total_skip",
            "IA_TOP",
            "IA_LEFT",
            "normalized_ID",
            "start_of_line",
            "end_of_line",
            # "gpt2_Surprisal",
            # "Wordfreq_Frequency",
            # "Length",
            # "Is_Content_Word",
            # "n_Lefts",
            # "n_Rights",
            # "Distance2Head",
        ]
    )

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
        ]
    )


@register_config(group=GROUP)
@dataclass
class PostFusionSelectedAnswersMultiClass(PostFusion):
    prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
    preorder: bool = False


@register_config(group=GROUP)
@dataclass
class PostFusionFreeze(PostFusion):
    freeze: bool = True