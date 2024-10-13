""" Configuration file for the model, trainer, data paths and data. """ ""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.base_model_args import BaseModelArgs
from src.configs.model_args.base_model_args_ml import BaseMLModelArgs
from src.configs.trainer_args import Base
from src.configs.trainer_args_ml import BaseMLTrainerArgs
from src.models.ahn_baseline import AhnCNNModel, AhnRNNModel
from src.models.base_model import BaseModel
from src.models.models_ml import BaseMLModel
from src.models.beyelstm_model import BEyeLSTMModel
from src.models.eyettention_model import EyettentionModel
from src.models.fse_model import FSEModel
from src.models.lit_model import LitModel
from src.models.mag_model import MAGModel
from src.models.mlp_model import MLPModel
from src.models.models_ml import (
    DummyClassifierMLModel,
    KNearestNeighborsMLModel,
    LogisticRegressionMLModel,
    SupportVectorMachineMLModel,
)
from src.models.post_fusion_model import PostFusionModel
from src.models.roberteye_model import RoBERTeyeModel
from src.models.total_rt_mlp import TotalRtMLP


@dataclass
class Args:
    data_path: DataPathArgs = field(default_factory=DataPathArgs)
    model: BaseModelArgs = field(default_factory=BaseModelArgs)
    trainer: Base = field(default_factory=Base)
    data: DataArgs = field(default_factory=DataArgs)
    eval_path: str | None = None
    # https://hydra.cc/docs/1.3/configure_hydra/workdir/
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "outputs/${hydra:job.override_dirname}/fold_index=${data.fold_index}"
            },
            "sweep": {
                "dir": "cross_validation_runs",
                # https://github.com/facebookresearch/hydra/issues/1786#issuecomment-1017005470
                "subdir": "${hydra:job.override_dirname}/fold_index=${data.fold_index}",
            },
            "job": {
                "config": {
                    "override_dirname": {
                        # Don't include fold_index and devices in the directory name
                        "exclude_keys": ["data.fold_index", "trainer.devices"]
                    }
                }
            },
        }
    )


@dataclass
class MLArgs:
    data_path: DataPathArgs = field(default_factory=DataPathArgs)
    model: BaseMLModelArgs = field(default_factory=BaseMLModelArgs)
    trainer: BaseMLTrainerArgs = field(default_factory=BaseMLTrainerArgs)
    data: DataArgs = field(default_factory=DataArgs)
    eval_path: str | None = None
    # https://hydra.cc/docs/1.3/configure_hydra/workdir/
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "outputs/${hydra:job.override_dirname}/fold_index=${data.fold_index}"
            },
            "sweep": {
                "dir": "cross_validation_runs",
                # https://github.com/facebookresearch/hydra/issues/1786#issuecomment-1017005470
                "subdir": "${hydra:job.override_dirname}/fold_index=${data.fold_index}",
            },
            "job": {
                "config": {
                    "override_dirname": {
                        # Don't include fold_index and devices in the directory name
                        "exclude_keys": ["data.fold_index", "trainer.devices"]
                    }
                }
            },
        }
    )


class ModelMapping(Enum):
    """
    Enum for mapping model names to model classes.
    """

    LIT_MODEL = LitModel
    MLP_MODEL = MLPModel
    MAG_MODEL = MAGModel
    FSE_MODEL = FSEModel
    TOTAL_RT_MPL_MODEL = TotalRtMLP
    ROBERTEYE_MODEL = RoBERTeyeModel
    AHN_CNN_MODEL = AhnCNNModel
    AHN_RNN_MODEL = AhnRNNModel
    BEYELSTM_MODEL = BEyeLSTMModel
    EYETTENTION_MODEL = EyettentionModel
    POSTFUSION_MODEL = PostFusionModel

    # ML models
    LOGISTIC_REGRESSION = LogisticRegressionMLModel
    KNN = KNearestNeighborsMLModel
    DUMMY_CLASSIFIER = DummyClassifierMLModel
    SVM = SupportVectorMachineMLModel


def get_model(cfg: Args) -> BaseModel:
    """
    Returns a model based on the model name.
    """
    model_class = ModelMapping[cfg.model.model_params.model_name].value
    return model_class(
        trainer_args=cfg.trainer,
        model_args=cfg.model,  # type: ignore
        data_args=cfg.data,  # type: ignore
        data_path_args=cfg.data_path,  # type: ignore
    )


def get_model_ml(
    trainer_args: BaseMLTrainerArgs, model_args: BaseMLModelArgs
) -> BaseMLModel:
    """
    # TODO send data_args and data_path_args like get_model?
    Returns a model based on the model name.
    """
    model_class = ModelMapping[model_args.model_params.model_name].value
    return model_class(trainer_args=trainer_args, model_args=model_args)  # type: ignore


def move_target_column_to_end(cfg: Args | MLArgs):
    if cfg.model.prediction_config.target_column in cfg.data.groupby_columns:
        # put cfg.model.target_column at the end of the list (important for the line in datasets:
        # `labels = [ key_[-1] for key_ in self.ordered_key_list]`)
        cfg.data.groupby_columns.remove(cfg.model.prediction_config.target_column)
    cfg.data.groupby_columns.append(cfg.model.prediction_config.target_column)

    return cfg
