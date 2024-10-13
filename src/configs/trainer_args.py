"""
This module contains dataclasses for configuring the PyTorch Lightning trainer.

The module defines a base configuration class `Base` and several derived classes
for specific models such as Shubi, Ahn, BEyeLSTM, Yoav, and Eyettention.

Each configuration class is defined using the `@dataclass` decorator and specifies
the relevant attributes and their default values for the corresponding model's trainer.

The configuration classes use constants from the `src.configs.constants` module, such as
`Accelerators`, `ConfigName`, `MatmulPrecisionLevel`, `Precision`, and `RunModes`.

The `@register_config` decorator is used to register the configuration classes with a
specific group defined by `ConfigName.TRAINER`.
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from src.configs.constants import (
    Accelerators,
    ConfigName,
    MatmulPrecisionLevel,
    Precision,
    RunModes,
)
from src.configs.utils import register_config

GROUP = ConfigName.TRAINER


@dataclass
class Base:
    """
    Base configuration class for the PyTorch Lightning trainer.

    This class defines common attributes shared by all trainer configurations.

    Attributes:
        num_workers (int): Number of worker processes for data loading. Default is 10.
        warmup_proportion (float | None): Proportion of training steps for learning rate warmup. Default is None.
        max_epochs (int): Maximum number of training epochs. Must be specified by derived classes.
        profiler (str | None): Profiler to use ('simple', 'advanced', or None). Default is None.
        weight_decay (float | None): Weight decay factor for regularization. Default is None. Put zero for no weight decay in models that expect it.
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 10.
        learning_rate (float): Learning rate for the optimizer. Must be specified by derived classes.
        gradient_clip_val (float | None): Gradient clipping value. Default is None.
        accelerator (Accelerators): Accelerator to use (e.g., 'cpu', 'gpu', 'tpu'). Default is Accelerators.AUTO.
        log_gradients (bool): Whether to log gradients. Default is False.
        precision (Precision): Numerical precision for training. Default is Precision.THIRTY_TWO_TRUE.
        float32_matmul_precision (MatmulPrecisionLevel): Matrix multiplication precision level. Default is MatmulPrecisionLevel.HIGH.
        seed (int): Random seed for reproducibility. Default is 42.
        devices (list[int]): List of device IDs to use for training. Default is [0].
        run_mode (RunModes): Mode for running the trainer (e.g., 'train', 'test', 'debug'). Default is RunModes.TRAIN.
        wandb_job_type (str): Type of job for Weights & Biases logging. Default is "MISSING".
        wandb_project (str): Weights & Biases project name. Default is "reading-comprehension-from-eye-movements".
        wandb_entity (str): Weights & Biases entity name. Default is "lacc-lab".
        wandb_notes (str): Additional notes for Weights & Biases logging. Default is an empty string.
        max_time (None | str): Maximum time allowed for training. Default is None (no limit). # Interrupts at the *end* of an epoch if duration is larger than max_time.
    """

    num_workers: int = 10
    warmup_proportion: float | None = None  #! not implemented in all models
    max_epochs: int = MISSING
    profiler: str | None = None
    weight_decay: float | None = None  #! not implemented in all models
    early_stopping_patience: int = 10
    learning_rate: float = MISSING
    gradient_clip_val: float | None = None
    accelerator: Accelerators = Accelerators.AUTO
    log_gradients: bool = False
    precision: Precision = Precision.THIRTY_TWO_TRUE
    float32_matmul_precision: MatmulPrecisionLevel = MatmulPrecisionLevel.HIGH
    seed: int = 42  # TODO save folds separately if changing the seed!!!
    # Note, metric logging is not fully supported for num_devices > 1
    # (if I remember correctly confusion matrix is the issue) and slows the run *considerably*.
    # Also note that samples can be duplicated and thus the metrics are not fully accurate.
    devices: Any = 1
    run_mode: RunModes = RunModes.TRAIN
    wandb_job_type: str = "MISSING"
    wandb_project: str = "reading-comprehension-from-eye-movements"
    wandb_entity: str = "lacc-lab"
    wandb_notes: str = ""
    do_fancy_sampling: bool = False  # TODO add to docs
    samples_per_epoch: int = -1  # Only used when do_fancy_sampling is True
    max_time: None | str = (
        None  # | timedelta | dict[str, int] # Trainer supports these as well but hydra/omegaconf don't support it in typing.
        # https://hydra.cc/docs/tutorials/structured_config/intro/#structured-configs-limitations
    )

    def __post_init__(self):
        """
        Post-initialization hook to adjust attributes based on the run mode.

        If the run mode is set to 'debug', the number of workers is set to 0
        and the Weights & Biases job type is set to "debug".
        """
        if self.run_mode == RunModes.DEBUG:
            self.num_workers = 0
            self.wandb_job_type = "debug"


@register_config(group=GROUP)
@dataclass
class Shubi(Base):
    """
    Configuration for Shubi's trainer.

    Inherits from Base.

    Attributes:
        num_workers (int): Number of worker processes for data loading. Default is 6.
        max_epochs (int): Maximum number of training epochs. Default is 10.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-5.
        devices (list[int]): List of device IDs to use for training. Default is [1].
        warmup_proportion (float): Proportion of training steps for learning rate warmup. Default is 0.06.
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 5.
    """

    num_workers: int = 10
    max_epochs: int = 10
    learning_rate: float = 1e-5
    accelerator: Accelerators = Accelerators.GPU
    warmup_proportion: float = 0.06
    early_stopping_patience: int = 5
    do_fancy_sampling: bool = True


@register_config(group=GROUP)
@dataclass
class ReadingCompBase(Base):
    """
    Configuration for Shubi's trainer.

    Inherits from Base.

    Attributes:
        num_workers (int): Number of worker processes for data loading. Default is 6.
        max_epochs (int): Maximum number of training epochs. Default is 10.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-5.
        devices (list[int]): List of device IDs to use for training. Default is [1].
        warmup_proportion (float): Proportion of training steps for learning rate warmup. Default is 0.06.
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 5.
    """

    num_workers: int = 8
    max_epochs: int = 60
    learning_rate: float = 1e-5
    warmup_proportion: float = 0.1
    early_stopping_patience: int = 12
    do_fancy_sampling: bool = False


@register_config(group=GROUP)
@dataclass
class IsCorrectSampling(ReadingCompBase):
    do_fancy_sampling: bool = True
    samples_per_epoch: int = (
        42_800 // 6
    )  # 42.8K corresponds to <1% not picking a specific sample given 4636 "A" samples.


@register_config(group=GROUP)
@dataclass
class Ahn(IsCorrectSampling):
    """
    Configuration for Ahn's trainer.

    Inherits from Base.

    Attributes:
        max_epochs (int): Maximum number of training epochs. Default is 1000.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        devices (list[int]): List of device IDs to use for training. Default is [2].
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 50.
    """

    max_epochs: int = 1000
    learning_rate: float = 0.001
    early_stopping_patience: int = 50


@register_config(group=GROUP)
@dataclass
class BEyeLSTM(Base):
    """
    Configuration for BEyeLSTM trainer.

    Inherits from Base.

    Attributes:
        max_epochs (int): Maximum number of training epochs. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 20.
        gradient_clip_val (float): Gradient clipping value. Default is 1.0.
    """

    max_epochs: int = 100
    accelerator: Accelerators = Accelerators.GPU
    learning_rate: float = 0.001
    early_stopping_patience: int = 30
    samples_per_epoch: int = (
        42_800 // 6
    )  # 42.8K corresponds to <1% not picking a specific sample given 4636 "A" samples.

    do_fancy_sampling: bool = True
    num_workers: int = 10


@register_config(group=GROUP)
@dataclass
class Yoav(Base):
    """
    Configuration for Yoav's trainer.

    Inherits from Base.

    Attributes:
        max_epochs (int): Maximum number of training epochs. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 0.00003.
        devices (list[int]): List of device IDs to use for training. Default is [1].
        weight_decay (float): Weight decay factor for regularization. Default is 1e-4.
    """

    max_epochs: int = 100
    learning_rate: float = 0.00003
    devices: list[int] = field(default_factory=lambda: [1])
    weight_decay: float = 1e-4


@register_config(group=GROUP)
@dataclass
class Eyettention(IsCorrectSampling):
    """
    Configuration for Eyettention trainer.

    Inherits from Base.

    Attributes:
        max_epochs (int): Maximum number of training epochs. Default is 20.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 20.
    """

    learning_rate: float = 1e-3


## Cfir's Trainers ##
@register_config(group=GROUP)
@dataclass
class CfirRoBERTaEye(Base):
    """
    Configuration for Shubi's trainer.

    Inherits from Base.

    Attributes:
        num_workers (int): Number of worker processes for data loading. Default is 6.
        max_epochs (int): Maximum number of training epochs. Default is 10.
        learning_rate (float): Learning rate for the optimizer. Default is 1e-5.
        devices (list[int]): List of device IDs to use for training. Default is [1].
        warmup_proportion (float): Proportion of training steps for learning rate warmup. Default is 0.06.
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 5.
    """

    num_workers: int = 10
    max_epochs: int = 10
    learning_rate: float = 1e-5
    warmup_proportion: float = 0.06
    early_stopping_patience: int = 5
    do_fancy_sampling: bool = False


@register_config(group=GROUP)
@dataclass
class CfirBEyeLSTM(Base):
    """
    Configuration for BEyeLSTM trainer.

    Inherits from Base.

    Attributes:
        max_epochs (int): Maximum number of training epochs. Default is 100.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 20.
        gradient_clip_val (float): Gradient clipping value. Default is 1.0.
    """

    max_epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 20
    gradient_clip_val: float = 1.0


@register_config(group=GROUP)
@dataclass
class CfirMAG(Base):
    """
    Configuration for MAG trainer.

    Inherits from Base.

    Attributes:
        max_epochs (int): Maximum number of training epochs. Default is 1000.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        devices (list[int]): List of device IDs to use for training. Default is [2].
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping. Default is 50.
    """

    max_epochs: int = 10
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    warmup_proportion: float = 0.06
