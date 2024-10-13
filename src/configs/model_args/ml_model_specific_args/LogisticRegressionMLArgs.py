from dataclasses import dataclass, field


from src.configs.constants import (
    MLModelNames,
    ConfigName,
    ItemLevelFeaturesModes,
)
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
class LogisticRegressionMLParams(BaseMLModelParams):
    concat_or_duplicate: str = "concat"
    item_level_features_mode: ItemLevelFeaturesModes = ItemLevelFeaturesModes.NONE

    model_name: MLModelNames = MLModelNames.LOGISTIC_REGRESSION
    sklearn_pipeline: tuple = (
        ("scaler", "sklearn.preprocessing.StandardScaler"),
        ("clf", "sklearn.linear_model.LogisticRegression"),
    )
    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # clf params
    sklearn_pipeline_param_clf__C: float = 2.0
    sklearn_pipeline_param_clf__fit_intercept: bool = True
    sklearn_pipeline_param_clf__penalty: str = "l2"
    sklearn_pipeline_param_clf__solver: str = "lbfgs"
    sklearn_pipeline_param_clf__random_state: int = 1
    sklearn_pipeline_param_clf__max_iter: int = 1000
    sklearn_pipeline_param_clf__class_weight: str = "balanced"

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True


@register_config(group=GROUP)
@dataclass
class LogisticRegressionMLArgs(BaseMLModelArgs):

    batch_size: int = -1
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams()
    )
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    #! note logistic regression is for binary classification
    text_dim: int = 768  # ?
    max_seq_len: int = 303  # longest text input sequence in the dataset in tokens #?
    max_eye_len: int = 300  # longest eye sequence in the dataset in tokens #?
    use_fixation_report: bool = True
    backbone: str = "roberta-base"


# Condition Prediction
class LogisticRegressionCondPredMLArgs(LogisticRegressionMLArgs):
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)


@register_config(group=GROUP)
@dataclass
class LogisticRegressionCondPredDavidILFMLArgs(LogisticRegressionCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DAVID
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionCondPredLennaILFMLArgs(LogisticRegressionCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.LENNA
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionCondPredDianeILFMLArgs(LogisticRegressionCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DIANE
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionCondPredReadingTimeMLArgs(LogisticRegressionCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.TOTAL_READING_TIME
        )
    )


# IsCorrect Prediction Logistic Regression
@register_config(group=GROUP)
@dataclass
class LogisticRegressionIsCorrectPredMLArgs(LogisticRegressionMLArgs):
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
    max_seq_len: int = 311


@register_config(group=GROUP)
@dataclass
class LogisticRegressionIsCorrectPredDavidILFMLArgs(
    LogisticRegressionIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DAVID
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionIsCorrectPredLennaILFMLArgs(
    LogisticRegressionIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.LENNA
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionIsCorrectPredDianeILFMLArgs(
    LogisticRegressionIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DIANE
        )
    )


@register_config(group=GROUP)
@dataclass
class LogisticRegressionIsCorrectPredReadingTimeMLArgs(
    LogisticRegressionIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.TOTAL_READING_TIME
        )
    )


# Question Prediction Logistic Regression
@register_config(group=GROUP)
@dataclass
class LogisticRegressionQPredMLArgs(LogisticRegressionMLArgs):
    prediction_config: PredCfg = field(default_factory=QPredCfg)
    model_params: BaseMLModelParams = field(
        default_factory=lambda: LogisticRegressionMLParams(
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
