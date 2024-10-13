from dataclasses import dataclass
from typing import Any


from src.configs.constants import (
    ConfigName,
    MatmulPrecisionLevel,
    Precision,
    RunModes,
)
from src.configs.utils import register_config

GROUP = ConfigName.TRAINER


@dataclass
class BaseMLTrainerArgs:
    """
    Trainer arguments for the machine learning models.
    """

    num_workers: int = 10
    profiler: str | None = None  # 'simple' | 'advanced' | None  #! Not supported
    precision: Precision = Precision.THIRTY_TWO_TRUE  #! Not supported
    float32_matmul_precision: MatmulPrecisionLevel = (
        MatmulPrecisionLevel.HIGH
    )  #! Supported for torch matrices only
    seed: int = 42
    devices: Any = (
        1  # Number of GPUs to use  #! set CUDA_VISIBLE_DEVICES to control which GPUs are used
    )
    run_mode: RunModes = RunModes.TRAIN
    wandb_job_type: str = (
        "MISSING"  # TODO CHANGE TO MISSING BETWEEN RUNS  # * Try to stick to 'debug', 'cv', single
    )
    wandb_project: str = "MISSING"
    wandb_entity: str = "lacc-lab"
    wandb_notes: str = ""
    max_time: None | str = (
        None  # Interupts at the *end* of an epoch if duration is larger than max_time.  #! Not supported
    )
    batch_size: int = 32  # used for batch encoding for certain features

    def __post_init__(self):
        if self.run_mode == RunModes.DEBUG:
            self.num_workers = 0
            self.wandb_job_type = "debug"


@register_config(group=GROUP)
@dataclass
class CfirMLQCond(BaseMLTrainerArgs):
    """
    Trainer arguments for the Cfir model.
    """

    wandb_project: str = "question_decoding+gathering"


@register_config(group=GROUP)
@dataclass
class CfirMLQPred(BaseMLTrainerArgs):
    """
    Trainer arguments for the Cfir model.
    """

    wandb_project: str = "question_decoding"


@register_config(group=GROUP)
@dataclass
class CfirJunk(BaseMLTrainerArgs):
    """
    Trainer arguments for the Cfir model.
    """

    wandb_project: str = "junk"
