"""Main file for cognitive state decoding training"""

import pprint
import dataclasses

import hydra
import lightning_fabric as lf
import torch
import wandb
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from src import datamodule
from src.configs.constants import RunModes
from src.configs.main_config import (
    MLArgs,
    get_model_ml,
    move_target_column_to_end,
)
from src.configs.model_args.base_model_args_ml import BaseMLModelArgs
from src.configs.trainer_args_ml import BaseMLTrainerArgs
from src.train_utils import (
    update_cfg_with_wandb,
)

cs = ConfigStore.instance()
cs.store(name="config", node=MLArgs)


@hydra.main(version_base=None, config_name="config")
def main(cfg: MLArgs) -> None:
    # instantiate the config as primitive types
    cfg = instantiate(config=cfg, _convert_="object")
    lf.seed_everything(seed=cfg.trainer.seed)
    torch.set_float32_matmul_precision(precision=cfg.trainer.float32_matmul_precision)

    cfg = move_target_column_to_end(cfg=cfg)  # type: ignore
    assert isinstance(cfg, MLArgs)

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

    if wandb.run and wandb.config.as_dict():
        cfg = update_cfg_with_wandb(cfg)  # type: ignore
    pprint.pprint(cfg)

    # convert cfg to dictionary
    cfg_dict = dataclasses.asdict(cfg)

    # log the config to wandb
    if cfg.trainer.run_mode != RunModes.FAST_DEV_RUN:
        wandb.config.update(cfg_dict)

    dm = datamodule.ETDataModuleFast(cfg=cfg)
    dm.prepare_data()
    dm.setup(stage="fit")

    if cfg.model.model_params.class_weights is not None:
        class_weights = dm.train_dataset.ordered_label_counts["count"]
        cfg.model.model_params.class_weights = (
            sum(class_weights) / class_weights
        ).tolist()
        print(f"Class weights: {cfg.model.model_params.class_weights}")

    assert isinstance(cfg.trainer, BaseMLTrainerArgs)
    assert isinstance(cfg.model, BaseMLModelArgs)
    model = get_model_ml(trainer_args=cfg.trainer, model_args=cfg.model)

    model.fit(dm=dm)

    # evaluate the model on train
    model.evaluate(eval_dataset=dm.train_dataset, stage="train", validation_map="all")
    model.on_stage_end()

    # evaluate the model on val
    for validation_map, val_dataset in zip(
        model.val_maps, dm.val_datasets
    ):  #! drops the 'all' validation map as datasets don't include it
        model.evaluate(
            eval_dataset=val_dataset, stage="val", validation_map=validation_map
        )

    model.on_stage_end()

    # ? dm.setup(stage="test")
    # ? model.evaluate(eval_datasets=dm.test_datasets, stage="test")

    if cfg.trainer.run_mode != RunModes.FAST_DEV_RUN:
        wandb.finish()


if __name__ == "__main__":
    main()
