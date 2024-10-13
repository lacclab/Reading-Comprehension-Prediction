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
class DummyClassifierMLParams(BaseMLModelParams):
    concat_or_duplicate: str = "concat"
    item_level_features_mode: ItemLevelFeaturesModes = ItemLevelFeaturesModes.NONE
    model_name: MLModelNames = MLModelNames.DUMMY_CLASSIFIER
    sklearn_pipeline: tuple = (("clf", "sklearn.dummy.DummyClassifier"),)
    # sklearn pipeline params
    #! note the naming convention for the parameters:
    #! sklearn_pipeline_param_<pipline_element_name>__<param_name>

    # clf params
    sklearn_pipeline_param_clf__strategy: str = (
        "most_frequent"  # "stratified", "most_frequent", "prior", "uniform"
    )
    sklearn_pipeline_param_clf__random_state: int = 1


@register_config(group=GROUP)
@dataclass
class DummyClassifierMLArgs(BaseMLModelArgs):
    batch_size: int = -1
    model_params: BaseMLModelParams = field(
        default_factory=lambda: DummyClassifierMLParams()
    )
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)
    text_dim: int = 768  # ?
    max_seq_len: int = 303  # longest text input sequence in the dataset in tokens #?
    max_eye_len: int = 300  # longest eye sequence in the dataset in tokens #?
    use_fixation_report: bool = True
    backbone: str = "roberta-base"


@register_config(group=GROUP)
@dataclass
class DummyClassifierCondPredMLArgs(DummyClassifierMLArgs):
    prediction_config: PredCfg = field(default_factory=ConditionPredCfg)


@register_config(group=GROUP)
@dataclass
class DummyClassifierIsCorrectPredMLArgs(DummyClassifierMLArgs):
    prediction_config: PredCfg = field(default_factory=IsCorrectPredCfg)
    max_seq_len: int = 311


@register_config(group=GROUP)
@dataclass
class DummyClassifierQPredMLArgs(DummyClassifierMLArgs):
    prediction_config: PredCfg = field(default_factory=QPredCfg)
    model_params: BaseMLModelParams = field(
        default_factory=lambda: DummyClassifierMLParams(
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
