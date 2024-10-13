from dataclasses import dataclass, field

from src.configs.constants import (
    BackboneNames,
    ConfigName,
    DataRepresentation,
    ModelNames,
    FeatureMode,
)
from src.configs.model_args.base_model_args import BaseModelArgs, BaseModelParams
from src.configs.prediction_modes import (
    # ChosenAnswerPredCfg,
    # ConditionPredCfg,
    ChosenAnswerPredCfg,
    IsCorrectPredCfg,
    PredCfg,
    # QConditionPredCfg,
    # QPredCfg,
)
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class MagParams(BaseModelParams):
    """
    Configuration for the Mag model.

    Attributes
    ----------
    mag_dropout : float
        Dropout rate for the MAG module.
    mag_beta_shift : float
        Beta in MAG module.
    concat_or_duplicate : MagModelModes
        The mode for handling the multiple-choice data.
        This can be either 'concat' to concatenate the data, or 'duplicate' to duplicate the data.
    """

    model_name: ModelNames = ModelNames.MAG_MODEL
    mag_dropout: float = 0.5
    mag_beta_shift: float = 1
    concat_or_duplicate: DataRepresentation = DataRepresentation.CONCAT
    mag_injection_index: int = 1


# @register_config(group=GROUP)
# @dataclass
# class MAGArgs(BaseModelArgs):
#     """
#     max_seq_len and max_eye_len must have the same length.
#     """

#     prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=[1.0],  # overwritten by trainer
#         )
#     )
#     use_fixation_report: bool = False
#     add_contrastive_loss: bool = False
#     backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
#     max_seq_len: int = 257
#     max_eye_len: int = 257
#     batch_size: int = 4
#     # See https://www.semanticscholar.org/reader/077f8329a7b6fa3b7c877a57b81eb6c18b5f87de
#     accumulate_grad_batches: int = 4


# @register_config(group=GROUP)
# @dataclass
# class MAGACLArgs(MAGArgs):
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=None,
#         )
#     )
#     add_contrastive_loss: bool = True


# @register_config(group=GROUP)
# @dataclass
# class MAGCondPredConcatNoFix(MAGArgs):
#     prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=None,
#         )
#     )
#     max_seq_len: int = 300
#     max_eye_len: int = 300
#     batch_size: int = 4
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class MAGQPredConcatNoFix(MAGArgs):
#     """
#     MAG model for question prediction with concatenated data representation.
#     """

#     prediction_config: PredCfg = field(default_factory=QPredCfg)
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=None,
#         )
#     )
#     max_seq_len: int = 300
#     max_eye_len: int = 300
#     batch_size: int = 4
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class MAGQPredDuplicateNoFix(MAGArgs):
#     """
#     MAG model for question prediction with concatenated data representation.
#     """

#     prediction_config: PredCfg = field(default_factory=QPredCfg)
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=None,
#         )
#     )
#     max_seq_len: int = 300
#     max_eye_len: int = 300
#     batch_size: int = 2
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class MAGQCondPredConcatNoFix(MAGArgs):
#     """
#     MAG model for question prediction with concatenated data representation.
#     """

#     prediction_config: PredCfg = field(default_factory=QConditionPredCfg)
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=None,
#         )
#     )
#     max_seq_len: int = 300
#     max_eye_len: int = 300
#     batch_size: int = 6
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class MAGQCondPredDuplicateNoFix(MAGArgs):
#     """
#     MAG model for question prediction with concatenated data representation.
#     """

#     prediction_config: PredCfg = field(default_factory=QConditionPredCfg)
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=None,
#         )
#     )
#     max_seq_len: int = 300
#     max_eye_len: int = 300
#     batch_size: int = 1
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @dataclass
# class MAGDuplicate(MAGArgs):
#     """MAG model with Duplicate data representation."""

#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=[1.0],  # Gets overritten by the trainer
#         )
#     )
#     max_seq_len: int = 257
#     max_eye_len: int = 257
#     batch_size: int = 8
#     # See https://www.semanticscholar.org/reader/077f8329a7b6fa3b7c877a57b81eb6c18b5f87de
#     accumulate_grad_batches: int = 2
#     backbone: BackboneNames = BackboneNames.ROBERTA_RACE


# ########### Other MAG Models #############


# @dataclass
# class MAGConcat(MAGArgs):
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(concat_or_duplicate=DataRepresentation.CONCAT)
#     )
#     max_seq_len: int = 300
#     max_eye_len: int = 300
#     batch_size: int = 64


# @dataclass
# class MAGDuplicateWeighted(MAGArgs):
#     """
#     MAGLargeDuplicate with class weights.
#     """

#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE,
#             # class_weights=[100 / 86.9, 100 / 7.6, 100 / 4.0, 100 / 1.5],
#             class_weights=[1.0, 18.0, 23.0, 30.0],
#             # class_weights=[1.0, 1.5, 2.0, 2.5],
#         )
#     )
#     max_seq_len: int = 259
#     max_eye_len: int = 259
#     batch_size: int = 16


# @dataclass
# class MAGDuplicateQPred(MAGArgs):
#     """
#     MAGLargeDuplicate for question prediction.
#     """

#     prediction_config: PredCfg = field(default_factory=QPredCfg)
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE
#         )
#     )
#     max_seq_len: int = 243


# @dataclass
# class MAGConcatIsCorrect(MAGArgs):
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(concat_or_duplicate=DataRepresentation.CONCAT)
#     )
#     max_seq_len: int = 309
#     max_eye_len: int = 309
#     prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)


# @dataclass
# class MAGConcatCondition(MAGArgs):
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(concat_or_duplicate=DataRepresentation.CONCAT)
#     )
#     max_seq_len: int = 309
#     max_eye_len: int = 309
#     prediction_config: PredCfg = field(default_factory=ConditionPredCfg)


# @register_config(group=GROUP)
# @dataclass
# class MAGConcatReadingComp(MAGArgs):
#     model_params: MagParams = field(
#         default_factory=lambda: MagParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             mag_injection_index=0,
#             mag_beta_shift=1e-3,
#             mag_dropout=0.5,
#             class_weights=None,
#         )
#     )
#     add_contrastive_loss: bool = False


@register_config(group=GROUP)
@dataclass
class MAG(BaseModelArgs):
    model_params: MagParams = field(
        default_factory=lambda: MagParams(
            concat_or_duplicate=DataRepresentation.CONCAT,
            mag_injection_index=0,
            mag_beta_shift=1e-3,
            mag_dropout=0.5,
            class_weights=None,
        )
    )
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
    max_seq_len: int = 313
    max_eye_len: int = 313
    batch_size: int = 4
    accumulate_grad_batches: int = 16 // batch_size
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    use_fixation_report: bool = False
    add_contrastive_loss: bool = False
    freeze: bool = False


@register_config(group=GROUP)
@dataclass
class MAGSelectedAnswersMultiClass(MAG):
    prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
    preorder: bool = False


@register_config(group=GROUP)
@dataclass
class MAGBase(MAG):
    backbone: BackboneNames = BackboneNames.ROBERTA_BASE

    def __post_init__(self):
        super().__post_init__()


# Mag freeze
@register_config(group=GROUP)
@dataclass
class MAGFreeze(MAG):
    freeze: bool = True


@register_config(group=GROUP)
@dataclass
class MAGWords(MAG):
    feature_mode: FeatureMode = FeatureMode.WORDS

    def __post_init__(self):
        super().__post_init__()


@register_config(group=GROUP)
@dataclass
class MAGEyes(MAG):
    feature_mode: FeatureMode = FeatureMode.EYES

    def __post_init__(self):
        super().__post_init__()
        
@register_config(group=GROUP)
@dataclass
class MAGSelectedAnswersMultiClassLing(MAGSelectedAnswersMultiClass):
    
    feature_mode: FeatureMode = FeatureMode.WORDS

    def __post_init__(self):
        super().__post_init__()