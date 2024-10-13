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
class RoBERTEyeParams(BaseModelParams):
    """
    Configuration for the RoBERTEye model.

    Attributes
    ----------
    concat_or_duplicate : MagModelModes
        The mode for handling the multiple-choice data.
        This can be either 'concat' to concatenate the data, or 'duplicate' to duplicate the data.
    prepend_eye_data : bool
        A flag indicating whether to prepend the eye data to the input.
        If True, the eye data will be added at the beginning of the input (otherwise no eyes!)
    """

    model_name: ModelNames = ModelNames.ROBERTEYE_MODEL
    concat_or_duplicate: DataRepresentation = DataRepresentation.DUPLICATE
    prepend_eye_data: bool = True


# @dataclass
# class RoBERTeyeArgs(BaseModelArgs):
#     """RoBERTeye Model"""

#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE,
#             class_weights=None,  # Gets overwritten by the trainer
#         )
#     )
#     max_seq_len: int = 260
#     prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
#     max_eye_len: int = 165
#     eye_projection_dropout: float = 0.5
#     backbone: BackboneNames = BackboneNames.ROBERTA_RACE
#     freeze: bool = False
#     add_contrastive_loss: bool = False
#     #! Don't change the following
#     n_tokens: int = 0
#     eye_token_id: int = 0
#     sep_token_id: int = 0
#     is_training: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeDuplicate(RoBERTeyeArgs):
#     """RoBERTeye Model (duplicate)"""

#     batch_size: int = 4
#     accumulate_grad_batches: int = 4
#     use_fixation_report: bool = False
#     add_contrastive_loss: bool = True


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeDuplicateFixation(RoBERTeyeArgs):
#     """RoBERTeye Model (duplicate) with fixation data"""

#     use_fixation_report: bool = True
#     batch_size: int = 4
#     accumulate_grad_batches: int = 4
#     add_contrastive_loss: bool = True


# @register_config(group=GROUP)
# @dataclass
# class RoBERTaNoEyes(RoBERTeyeArgs):
#     """RoBERTeye Model without eyes at all"""

#     batch_size: int = 4
#     accumulate_grad_batches: int = 4
#     use_fixation_report: bool = False
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE,
#             class_weights=None,
#             prepend_eye_data=False,
#         )
#     )
#     max_seq_len: int = 257
#     max_eye_len: int = 257


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcat(RoBERTeyeArgs):
#     """RoBERTeye Model (concat)"""

#     max_seq_len: int = 302
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT
#         )
#     )
#     batch_size: int = 16
#     use_fixation_report: bool = False


# @dataclass
# # qpred
# class RoBERTeyeDuplicateQPred(RoBERTeyeDuplicate):
#     """RoBERTeye Model (duplicate) for question prediction"""

#     prediction_config: PredCfg = field(default_factory=QPredCfg)


# ############ Other RoBERTeye Models ############


# @dataclass
# class RoBERTeyeConcatClassPred(RoBERTeyeConcat):
#     """RoBERTeye Model (concat) without answers"""

#     prediction_config: PredCfg = field(
#         default_factory=lambda: ChosenAnswerPredCfg(add_answers=False)
#     )


# @dataclass
# class RoBERTeyeIsCorrect(RoBERTeyeArgs):
#     """RoBERTeye Model"""

#     model_params: RoBERTeyeParams = field(
#         default_factory=lambda: RoBERTeyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT, prepend_eye_data=True
#         )
#     )
#     max_seq_len: int = 302
#     prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeCondition(RoBERTeyeArgs):
#     """RoBERTeye Model"""

#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT, prepend_eye_data=True
#         )
#     )
#     max_seq_len: int = 302
#     prediction_config: PredCfg = field(default_factory=ConditionPredCfg)

#     batch_size: int = 4
#     accumulate_grad_batches: int = 8
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeCondPredConcat(RoBERTeyeArgs):
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT, prepend_eye_data=True
#         )
#     )

#     max_seq_len: int = 302
#     prediction_config: PredCfg = field(default_factory=ConditionPredCfg)

#     batch_size: int = 3
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeQCondConcat(RoBERTeyeArgs):
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT, prepend_eye_data=True
#         )
#     )

#     max_seq_len: int = 302
#     prediction_config: PredCfg = field(default_factory=QConditionPredCfg)

#     batch_size: int = 3
#     accumulate_grad_batches: int = 12
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeQCondDuplicate(RoBERTeyeArgs):
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE, prepend_eye_data=True
#         )
#     )

#     max_seq_len: int = 302
#     prediction_config: PredCfg = field(default_factory=QConditionPredCfg)

#     batch_size: int = 5
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeQPredConcatNoFix(RoBERTeyeArgs):
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT, prepend_eye_data=True
#         )
#     )

#     max_seq_len: int = 302
#     prediction_config: PredCfg = field(default_factory=QPredCfg)

#     batch_size: int = 3
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeQPredDuplicateNoFix(RoBERTeyeArgs):
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.DUPLICATE, prepend_eye_data=True
#         )
#     )

#     max_seq_len: int = 302
#     prediction_config: PredCfg = field(default_factory=QPredCfg)

#     batch_size: int = 6
#     accumulate_grad_batches: int = 36 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatIAReadingComp(RoBERTeyeArgs):
#     """RoBERTeye Model (concat)"""

#     max_seq_len: int = 313
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT
#         )
#     )
#     batch_size: int = 4
#     accumulate_grad_batches: int = 16 // batch_size
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatFixationReadingComp(RoBERTeyeArgs):
#     """RoBERTeye Model (concat)"""

#     max_seq_len: int = 313
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT
#         )
#     )
#     batch_size: int = 4
#     accumulate_grad_batches: int = 16 // batch_size
#     use_fixation_report: bool = True


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatIsCorrect(RoBERTeyeArgs):
#     max_seq_len: int = 313
#     max_eye_len: int = 257
#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             class_weights=[1.0],
#         )
#     )
#     batch_size: int = 4
#     accumulate_grad_batches: int = 16 // batch_size
#     use_fixation_report: bool = False
#     prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatIAIsCorrect(RoBERTeyeConcatIsCorrect):
#     """RoBERTeye Model (concat)"""

#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             class_weights=[1.0],
#         )
#     )
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatIAIsCorrectCL(RoBERTeyeConcatIsCorrect):
#     """RoBERTeye Model (concat)"""

#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             class_weights=None,
#         )
#     )
#     use_fixation_report: bool = False
#     add_contrastive_loss: bool = True


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatIAIsCorrectCLSampling(RoBERTeyeConcatIsCorrect):
#     """RoBERTeye Model (concat)"""

#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             class_weights=None,
#         )
#     )
#     use_fixation_report: bool = False
#     add_contrastive_loss: bool = True
#     batch_size: int = 4
#     accumulate_grad_batches: int = 16 // batch_size


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatNoEyesIsCorrect(RoBERTeyeConcatIsCorrect):
#     """RoBERTeye Model (concat)"""

#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             class_weights=[1.0],
#             prepend_eye_data=False,
#         )
#     )
#     use_fixation_report: bool = False


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatFixationIsCorrectCLSampling(RoBERTeyeConcatIsCorrect):
#     """RoBERTeye Model (concat)"""

#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             class_weights=None,
#         )
#     )
#     use_fixation_report: bool = True
#     add_contrastive_loss: bool = True
#     batch_size: int = 4
#     accumulate_grad_batches: int = 16 // batch_size


# @register_config(group=GROUP)
# @dataclass
# class RoBERTeyeConcatNoEyesIsCorrectCLSampling(RoBERTeyeConcatIsCorrect):
#     """RoBERTeye Model (concat)"""

#     model_params: RoBERTEyeParams = field(
#         default_factory=lambda: RoBERTEyeParams(
#             concat_or_duplicate=DataRepresentation.CONCAT,
#             class_weights=None,
#             prepend_eye_data=False,
#         )
#     )
#     use_fixation_report: bool = False
#     add_contrastive_loss: bool = True


@dataclass
class Roberteye(BaseModelArgs):
    model_params: RoBERTEyeParams = field(
        default_factory=lambda: RoBERTEyeParams(
            concat_or_duplicate=DataRepresentation.CONCAT,
            class_weights=None,
        )
    )
    max_seq_len: int = 313
    max_eye_len: int = 313
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
    batch_size: int = 4
    accumulate_grad_batches: int = 16 // batch_size
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    freeze: bool = False
    add_contrastive_loss: bool = False
    eye_projection_dropout: float = 0.1

    #! Don't change the following
    n_tokens: int = 0
    eye_token_id: int = 0
    sep_token_id: int = 0
    is_training: bool = False


@register_config(group=GROUP)
@dataclass
class RoberteyeFixation(Roberteye):
    """RoBERTeye Model (concat)"""

    use_fixation_report: bool = True
    batch_size: int = 2
    accumulate_grad_batches: int = 16 // batch_size


@register_config(group=GROUP)
@dataclass
class RoberteyeWord(Roberteye):
    """RoBERTeye Model (concat)"""

    use_fixation_report: bool = False


@register_config(group=GROUP)
@dataclass
class RoberteyeWordLing(RoberteyeWord):
    """RoBERTeye Model (concat) without eye-tracking features"""

    feature_mode: FeatureMode = FeatureMode.WORDS

    def __post_init__(self):
        super().__post_init__()


@register_config(group=GROUP)
@dataclass
class RoberteyeWordEyes(RoberteyeWord):
    """RoBERTeye Model (concat) without eye-tracking features"""

    feature_mode: FeatureMode = FeatureMode.EYES

    def __post_init__(self):
        super().__post_init__()


@register_config(group=GROUP)
@dataclass
class Roberta(Roberteye):
    """RoBERTeye Model (concat)"""

    use_fixation_report: bool = False
    model_params: RoBERTEyeParams = field(
        default_factory=lambda: RoBERTEyeParams(
            concat_or_duplicate=DataRepresentation.CONCAT,
            class_weights=None,
            prepend_eye_data=False,
        )
    )


@register_config(group=GROUP)
@dataclass
class RobertaSelectedAnswersMultiClass(Roberta):
    prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
    preorder: bool = False


@register_config(group=GROUP)
@dataclass
class RoberteyeWordSelectedAnswersMultiClass(RoberteyeWord):
    prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
    preorder: bool = False
    max_seq_len: int = 314
    max_eye_len: int = 314


@register_config(group=GROUP)
@dataclass
class RoberteyeWordLingSelectedAnswersMultiClass(RoberteyeWord):
    prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
    preorder: bool = False

    feature_mode: FeatureMode = FeatureMode.WORDS

    def __post_init__(self):
        super().__post_init__()

    max_seq_len: int = 314
    max_eye_len: int = 314


@register_config(group=GROUP)
@dataclass
class RoberteyeFixationSelectedAnswersMultiClass(RoberteyeFixation):
    prediction_config: PredCfg = field(default_factory=ChosenAnswerPredCfg)
    preorder: bool = False
    max_seq_len: int = 314
    max_eye_len: int = 314
