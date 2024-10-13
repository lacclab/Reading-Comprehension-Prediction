from dataclasses import dataclass, field

from src.configs.constants import (
    MLModelNames,
    ConfigName,
    ItemLevelFeaturesModes,
)
from src.configs.model_args.base_model_args_ml import BaseMLModelArgs, BaseMLModelParams
from src.configs.prediction_modes import (
    IsCorrectPredCfg,
    PredCfg,
    ConditionPredCfg,
    QPredCfg,
)
from src.configs.utils import register_config


GROUP = ConfigName.MODEL


@dataclass
class KNearestNeighborsMLParams(BaseMLModelParams):
    """
    Configuration for the KNN model.
    * ILF stands for Item Level Features

    Attributes
    ----------
    concat_or_duplicate :
        The mode for handling the multiple-choice data.
        This can be either 'concat' to concatenate the data, or 'duplicate' to duplicate the data.
    prepend_eye_data : bool
        A flag indicating whether to prepend the eye data to the input.
        If True, the eye data will be added at the beginning of the input (otherwise no eyes!)
    """

    concat_or_duplicate: str = "concat"
    item_level_features_mode: ItemLevelFeaturesModes = ItemLevelFeaturesModes.NONE
    model_name: MLModelNames = MLModelNames.KNN
    sklearn_pipeline: tuple = (
        ("scaler", "sklearn.preprocessing.StandardScaler"),
        ("clf", "sklearn.neighbors.KNeighborsClassifier"),
    )
    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # clf params
    sklearn_pipeline_param_clf__n_neighbors: int = 5
    sklearn_pipeline_param_clf__weights: str = "distance"  # [uniform, distance]
    sklearn_pipeline_param_clf__algorithm: str = "auto"
    sklearn_pipeline_param_clf__leaf_size: int = 30
    sklearn_pipeline_param_clf__p: int = 2  # minkowski_distance (l_p) is used
    sklearn_pipeline_param_clf__metric: str = (
        "minkowski"  # shoudane use another metric.
    )

    # scaler params
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsMLArgs(BaseMLModelArgs):
    """
    Model arguments for the KNN model.
    """

    batch_size: int = -1
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams()
    )
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    text_dim: int = 768  # ?
    max_seq_len: int = 303  # longest text input sequence in the dataset in tokens #?
    max_eye_len: int = 300  # longest eye sequence in the dataset in tokens #?
    use_fixation_report: bool = True
    backbone: str = "roberta-base"


# Condition Prediction
@register_config(group=GROUP)
@dataclass
class KNearestNeighborsCondPredMLArgs(KNearestNeighborsMLArgs):
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsCondPredDavidILFMLArgs(KNearestNeighborsCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DAVID
        )
    )


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsCondPredLennaILFMLArgs(KNearestNeighborsCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.LENNA
        )
    )


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsCondPredDianeILFMLArgs(KNearestNeighborsCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DIANE
        )
    )


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsCondPredReadingTimeMLArgs(KNearestNeighborsCondPredMLArgs):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.TOTAL_READING_TIME
        )
    )


# IsCorrect Prediction
@register_config(group=GROUP)
@dataclass
class KNearestNeighborsIsCorrectPredMLArgs(KNearestNeighborsMLArgs):
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
    max_seq_len: int = 311


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsIsCorrectPredDavidILFMLArgs(
    KNearestNeighborsIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DAVID
        )
    )


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsIsCorrectPredLennaILFMLArgs(
    KNearestNeighborsIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.LENNA
        )
    )


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsIsCorrectPredDianeILFMLArgs(
    KNearestNeighborsIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.DIANE
        )
    )


@register_config(group=GROUP)
@dataclass
class KNearestNeighborsIsCorrectPredReadingTimeMLArgs(
    KNearestNeighborsIsCorrectPredMLArgs
):
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
            item_level_features_mode=ItemLevelFeaturesModes.TOTAL_READING_TIME
        )
    )


# Question Prediction
@register_config(group=GROUP)
@dataclass
class KNearestNeighborsQPredMLArgs(KNearestNeighborsMLArgs):
    prediction_config: PredCfg = field(default_factory=QPredCfg)
    model_params: BaseMLModelParams = field(
        default_factory=lambda: KNearestNeighborsMLParams(
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
