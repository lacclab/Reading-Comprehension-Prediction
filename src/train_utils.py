from datetime import timedelta
from pathlib import Path

import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger

from src.configs.constants import BackboneNames, Precision, RunModes
from src.configs.main_config import Args, MLArgs
from src.configs.trainer_args import Base


def update_cfg_with_wandb(cfg: Args | MLArgs) -> Args | MLArgs:
    print("Overwriting args with wandb config")
    for key, value in wandb.config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        print(
                            f"Setting cfg.{key}.{sub_key}.{sub_sub_key} to {sub_sub_value}"
                        )
                        setattr(
                            getattr(getattr(cfg, key), sub_key),
                            sub_sub_key,
                            sub_sub_value,
                        )
                else:
                    print(f"Setting cfg.{key}.{sub_key} to {sub_value}")
                    setattr(getattr(cfg, key), sub_key, sub_value)
        else:
            print(f"Setting cfg.{key} to {value}")
            setattr(cfg, key, value)

    cfg.model.text_dim = (
        768 if cfg.model.backbone == BackboneNames.ROBERTA_BASE else 1024
    )
    print(f"Setting cfg.model.text_dim to {cfg.model.text_dim}")

    return cfg


def setup_logger(
    wandb_project: str, wandb_entity: str, wandb_job_type: str
) -> WandbLogger:
    return WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        job_type=wandb_job_type,
    )


def setup_callbacks(
    dir_path: Path,
    early_stopping_patience: int,
    max_time: None | str | timedelta | dict[str, int],
) -> list[pl_callbacks.Callback]:
    monitor_metric = "Balanced_Accuracy/val_best_epoch_weighted_average"
    mode = "max"
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor=monitor_metric,
        mode=mode,
        filename="{epoch:02d}-highest_balanced_accuracy_val_weighted_average-{Balanced_Accuracy/val_best_epoch_weighted_average:.4f}",
        dirpath=dir_path,
        auto_insert_metric_name=False,
        verbose=True,
        enable_version_counter=False,
    )

    earlystopping = pl_callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=early_stopping_patience,
        mode=mode,
    )

    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval="step")

    model_summary = pl_callbacks.RichModelSummary(max_depth=4)

    timer = pl_callbacks.Timer(duration=max_time, interval="epoch")

    return [
        checkpoint_callback,
        earlystopping,
        lr_monitor,
        model_summary,
        pl_callbacks.RichProgressBar(),
        timer,
    ]


def configure_trainer(
    args: Base,
    callbacks: list,
    logger: WandbLogger | None = None,
    accumulate_grad_batches: int = 1,
) -> pl.Trainer:
    precision: Precision = args.precision
    log_every_n_steps = (
        1  # TODO this might be slowing down training, check if we can increase this
    )

    max_epochs = args.max_epochs
    devices = args.devices
    detect_anomaly = False
    limit_test_batches = None
    limit_val_batches = None
    limit_train_batches = None
    num_sanity_val_steps = 0
    fast_dev_run = False

    if args.run_mode == RunModes.DEBUG:
        max_epochs = 2
        limit_train_batches = 5
        limit_val_batches = 5
        limit_test_batches = 5
        num_sanity_val_steps = 2
        detect_anomaly = True
    elif args.run_mode == RunModes.FAST_DEV_RUN:
        fast_dev_run = True

    print(f"##### Running on devices: {devices} #####")
    print(f"##### Running with precision: {precision} #####")
    print(f"##### Running with max_epochs: {max_epochs} #####")

    return pl.Trainer(
        precision=precision,  # type: ignore
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator=args.accelerator,
        profiler=args.profiler,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        detect_anomaly=detect_anomaly,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        fast_dev_run=fast_dev_run,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
    )
