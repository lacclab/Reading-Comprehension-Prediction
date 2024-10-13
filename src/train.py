"""Main file for cognitive state decoding training"""

import pprint
from pathlib import Path

import hydra
import lightning_fabric as lf
import torch

# import torchinfo
import wandb
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from src.configs.model_args.base_model_args import BaseModelArgs
from src.configs.model_args.model_specific_args.MAGArgs import MagParams
from src.configs.trainer_args import Base

from src import datamodule
from src.configs.constants import BackboneNames, ModelNames, RunModes

# from src.configs.data_args import register_data_configs
from src.configs.main_config import (
    Args,
    get_model,
    move_target_column_to_end,
)
from src.train_utils import (
    configure_trainer,
    setup_callbacks,
    setup_logger,
    update_cfg_with_wandb,
)

cs = ConfigStore.instance()
cs.store(name="config", node=Args)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Args) -> None:
    try:
        import os

        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    except KeyError:
        print("CUDA_VISIBLE_DEVICES not set!")
    # instantiate the config as primitive types
    cfg = instantiate(config=cfg, _convert_="object")

    lf.seed_everything(seed=cfg.trainer.seed)
    torch.set_float32_matmul_precision(precision=cfg.trainer.float32_matmul_precision)

    cfg_ = move_target_column_to_end(cfg=cfg)
    # pylance workaround
    assert isinstance(cfg_, Args)
    cfg = cfg_
    
    # Fix model args numbers of features not updating when overwriting them any of them
    n_categorical_features = len(cfg.model.ia_categorical_features)
    cfg.model.eyes_dim = len(cfg.model.ia_features) - n_categorical_features
    cfg.model.fixation_dim = len(cfg.model.fixation_features) + cfg.model.eyes_dim
    cfg.model.ia_features_to_add_to_fixation_data = cfg.model.ia_features

    work_dir = HydraConfig.get().runtime.output_dir
    print(f"##### Work dir: {work_dir}")
    if cfg.trainer.run_mode != RunModes.FAST_DEV_RUN:
        wandb.init(
            entity=cfg.trainer.wandb_entity,
            project=cfg.trainer.wandb_project,
            job_type=cfg.trainer.wandb_job_type,
            notes=cfg.trainer.wandb_notes,
            dir=work_dir,
        )

        logger = setup_logger(
            wandb_entity=cfg.trainer.wandb_entity,
            wandb_project=cfg.trainer.wandb_project,
            wandb_job_type=cfg.trainer.wandb_job_type,
        )
    else:
        logger = None

    callbacks = setup_callbacks(
        dir_path=Path(work_dir),
        early_stopping_patience=cfg.trainer.early_stopping_patience,
        max_time=cfg.trainer.max_time,
    )

    # # If wandb config is not empty, we are running a sweep, so we need to set the args.
    if wandb.run and wandb.config.as_dict():
        cfg_ = update_cfg_with_wandb(cfg)
        # pylance workaround
        assert isinstance(cfg_, Args)
        cfg = cfg_

    pprint.pprint(cfg)

    assert isinstance(cfg.trainer, Base)
    assert isinstance(cfg.model, BaseModelArgs)
    if cfg.model.model_params.model_name == ModelNames.MAG_MODEL:
        assert cfg.model.max_eye_len == cfg.model.max_seq_len, (
            f"max_eye_len ({cfg.model.max_eye_len}) "
            f"and max_seq_len ({cfg.model.max_seq_len}) must have the same length for MAG."
        )

        assert isinstance(cfg.model.model_params, MagParams)
        if (
            cfg.model.backbone == BackboneNames.ROBERTA_BASE
            and cfg.model.model_params.mag_injection_index > 13
        ):
            print(
                f"Warning: MAG injection index {cfg.model.model_params.mag_injection_index} is higher than 13 for Roberta Base. Exiting."
            )
            wandb.finish(exit_code=1)

    trainer = configure_trainer(
        args=cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
    )

    dm = datamodule.ETDataModuleFast(cfg=cfg)
    dm.prepare_data()
    dm.setup(stage="fit")
    # Update class weights only if weighting
    if cfg.model.model_params.class_weights is not None:
        class_weights = dm.train_dataset.ordered_label_counts["count"]
        cfg.model.model_params.class_weights = (
            sum(class_weights) / class_weights
        ).tolist()
        print(f"Class weights: {cfg.model.model_params.class_weights}")

    if cfg.model.model_params.model_name in [
        ModelNames.ROBERTEYE_MODEL,
    ]:
        cfg.model.n_tokens = dm.train_dataset.n_tokens
        cfg.model.sep_token_id = dm.train_dataset.sep_token_id  # type: ignore
        cfg.model.eye_token_id = dm.train_dataset.eye_token_id
        cfg.model.is_training = True

    if cfg.model.model_params.model_name in [
        ModelNames.POSTFUSION_MODEL,
    ]:
        cfg.model.sep_token_id = dm.train_dataset.sep_token_id  # type: ignore

    model = get_model(cfg)
    # torchinfo.summary( # TODO doesn't work, try to fix?
    #     model=model,
    #     depth=4,
    #     input_size=[(16, 514), (16, 1028), (16, 1), (16, 514, 12), (16, 514)],
    #     dtypes=[torch.long, torch.long, torch.long, torch.float, torch.long],
    # )

    if logger is not None and cfg.trainer.log_gradients:
        logger.watch(model=model)

    # trainer.tune(model, datamodule=dm)

    trainer.fit(model=model, datamodule=dm)

    if trainer.checkpoint_callback:
        print(trainer.checkpoint_callback.best_model_path)  # type: ignore

    if cfg.trainer.run_mode != RunModes.FAST_DEV_RUN:
        wandb.finish()


if __name__ == "__main__":
    main()
