from dataclasses import dataclass, field


from src.configs.constants import MLModelNames, ConfigName, ItemLevelFeaturesModes
from src.configs.model_args.base_model_args_ml import BaseMLModelArgs, BaseMLModelParams
from src.configs.prediction_modes import (
    ConditionPredCfg,
    IsCorrectPredCfg,
    PredCfg,
    QPredCfg,
)
from src.configs.utils import register_config

GROUP = ConfigName.MODEL


@dataclass
class SupportVectorMachineMLParams(BaseMLModelParams):
    concat_or_duplicate: str = "concat"
    item_level_features_mode: ItemLevelFeaturesModes = ItemLevelFeaturesModes.NONE

    model_name: MLModelNames = MLModelNames.SVM
    sklearn_pipeline: tuple = (
        ("scaler", "sklearn.preprocessing.StandardScaler"),
        ("clf", "sklearn.svm.SVC"),
    )
    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # clf params
    sklearn_pipeline_params_clf__C: float = 1.0
    sklearn_pipeline_params_clf__kernel: str = "rbf"
    sklearn_pipeline_params_clf__degree: int = (
        3  # Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.
    )
    sklearn_pipeline_params_clf__gamma: str = "scale"
    sklearn_pipeline_params_clf__coef0: float = (
        0.0  # It is only significant in ‘poly’ and ‘sigmoid’.
    )
    sklearn_pipeline_params_clf__shrinking: bool = True
    sklearn_pipeline_params_clf__probability: bool = False
    sklearn_pipeline_params_clf__tol: float = 0.001
    sklearn_pipeline_params_clf__random_state: int = 1
    sklearn_pipeline_param_clf__class_weight: str = "balanced"

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineMLArgs(BaseMLModelArgs):
    batch_size: int = -1
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams()
    )
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    #! note logistic regression is for binary classification
    text_dim: int = 768  # ?
    max_seq_len: int = 303  # longest text input sequence in the dataset in tokens #?
    max_eye_len: int = 300  # longest eye sequence in the dataset in tokens #?
    use_fixation_report: bool = True
    backbone: str = "roberta-base"


# Question Prediction
# Condition Prediction
@register_config(group=GROUP)
@dataclass
class SupportVectorMachineQPredMLArgs(SupportVectorMachineMLArgs):
    prediction_config: PredCfg = field(default_factory=QPredCfg)
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            concat_or_duplicate="duplicate",
            use_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT=True,
            use_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT=True,
            item_level_features_mode=ItemLevelFeaturesModes.NONE,
        )
    )
    use_fixation_report: bool = False
    add_beyelstm_features: bool = False

    ia_features: list[str] = field(default_factory=lambda: ["IA_DWELL_TIME"])
    fixation_features: list[str] = field(default_factory=lambda: [])
    ia_categorical_features: list[str] = field(default_factory=lambda: [])


# Condition Prediction
@register_config(group=GROUP)
@dataclass
class SupportVectorMachineCondPredMLArgs(SupportVectorMachineMLArgs):
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineCondPredDavidILFMLArgs(SupportVectorMachineCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DAVID
        )
    )


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineCondPredLennaILFMLArgs(SupportVectorMachineCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.LENNA
        )
    )


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineCondPredDianeILFMLArgs(SupportVectorMachineCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DIANE
        )
    )


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineCondPredReadingTimeILFMLArgs(
    SupportVectorMachineCondPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.TOTAL_READING_TIME
        )
    )


# IsCorrect Prediction
@register_config(group=GROUP)
@dataclass
class SupportVectorMachineIsCorrectPredMLArgs(SupportVectorMachineMLArgs):
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
    max_seq_len: int = 311


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineIsCorrectPredDavidILFMLArgs(
    SupportVectorMachineIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DAVID
        )
    )


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineIsCorrectPredLennaILFMLArgs(
    SupportVectorMachineIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.LENNA
        )
    )


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineIsCorrectPredDianeILFMLArgs(
    SupportVectorMachineIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DIANE
        )
    )


@register_config(group=GROUP)
@dataclass
class SupportVectorMachineIsCorrectPredReadingTimeILFMLArgs(
    SupportVectorMachineIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: SupportVectorMachineMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.TOTAL_READING_TIME
        )
    )
