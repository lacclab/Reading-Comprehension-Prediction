"""Main file for testing cognitive state decoding models"""

import ast
import logging
import re
from collections import defaultdict
from os.path import join
from pathlib import Path

import hydra
import lightning_fabric as lf
import numpy as np
import pandas as pd
import torch
from hydra import compose
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from src.configs.main_config import Args, ModelMapping, move_target_column_to_end
from src.datamodule import ETDataModuleFast
from src.train_utils import configure_trainer


def create_and_configure_logger(log_name: str = "log.log") -> logging.Logger:
    """
    Creates and configures a logger
    Args:
        log_name (): The name of the log file
    Returns:
    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def convert_string_to_list(s: pd.Series) -> list:
    return s.apply(ast.literal_eval).tolist()


def defaultdict_to_df(
    macro_auroc: defaultdict, binary_auroc: defaultdict
) -> pd.DataFrame:
    # Convert the default dicts to DataFrames
    df_macro = pd.DataFrame(macro_auroc.items(), columns=["Eval Type", "AUROC"])
    df_binary = pd.DataFrame(binary_auroc.items(), columns=["Eval Type", "AUROC"])

    # Add a new column to distinguish between macro and binary
    df_macro["Task"] = "macro"
    df_binary["Task"] = "binary"

    # Concatenate the DataFrames
    df = pd.concat([df_macro, df_binary])

    # Calculate the average and standard deviation of AUROC
    df["Average AUROC"] = df["AUROC"].apply(np.mean)
    df["STD AUROC"] = df["AUROC"].apply(np.std)

    # # Set the index
    df.set_index(["Task", "Eval Type"], inplace=True)
    return df


def raw_res_to_auroc(res: pd.DataFrame) -> pd.DataFrame:
    grouped_res = res.groupby(["eval_type", "fold_index"])
    macro_auroc = defaultdict(list)
    binary_auroc = defaultdict(list)
    for (eval_type, fold_index), group_data in grouped_res:
        labels = group_data["label"].tolist()
        preds = convert_string_to_list(group_data["prediction_prob"])
        macro_auroc[eval_type].append(
            round(
                roc_auc_score(
                    y_true=labels, y_score=preds, average="macro", multi_class="ovr"
                ),
                3,
            )
        )

        binary_labels = group_data["binary_label"].tolist()
        binary_preds = group_data["binary_prediction_prob"].tolist()
        binary_auroc[eval_type].append(
            round(roc_auc_score(y_true=binary_labels, y_score=binary_preds), 3)
        )

    return defaultdict_to_df(macro_auroc, binary_auroc)


def get_config(config_path: Path) -> Args:
    """
    Load the config for testing.
    """
    output_dir = to_absolute_path(str(config_path))
    overrides = OmegaConf.load(join(output_dir, ".hydra/overrides.yaml"))

    hydra_config = OmegaConf.load(join(output_dir, ".hydra/hydra.yaml"))
    # getting the config name from the previous job.
    config_name = hydra_config.hydra.job.config_name
    # compose a new config from scratch
    cfg = compose(config_name, overrides=overrides)  # type: ignore
    updated_cfg = instantiate(cfg, _convert_="object")

    updated_cfg = move_target_column_to_end(updated_cfg)

    # pprint.pprint(updated_cfg)
    assert isinstance(updated_cfg, Args)
    return updated_cfg


def get_checkpoint_path(search_path, checkpoint_template, logger):
    checkpoint_files = list(search_path.glob(checkpoint_template))
    logger.info(f"Found {len(checkpoint_files)} checkpoints!")
    logger.info([f.name for f in checkpoint_files])
    # Extract version numbers and sort the list in descending order
    # this is a hacky way to get the version number from the file name
    # Extract highest_classless_accuracy_val_average- values and sort the list in descending order
    pattern = re.compile(
        r"highest_balanced_accuracy_val_weighted_average-(\d+\.\d+)(-v\d+)?\.ckpt$"
    )
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda f: float(
            re.search(
                pattern,
                str(f.name),
            ).group(1)  # type: ignore
            if re.search(
                pattern,
                str(f.name),
            )
            else 0.0
        ),
        reverse=True,
    )
    if not checkpoint_files:
        raise FileNotFoundError(
            "No checkpoint files found for pattern {checkpoint_template}!"
        )
    return checkpoint_files[0]


cs = ConfigStore.instance()
cs.store(name="config", node=Args)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Args) -> None:
    lf.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    assert cfg.eval_path is not None, "eval_path must be specified!"
    base_path = Path(cfg.eval_path)
    checkpoint_template = "*highest_balanced_accuracy_val_weighted_average*.ckpt"

    regime_names = [
        "new_item",
        "new_subject",
        "new_item_and_subject",
        # "all",
    ]  # This order is defined in the data module!
    group_level_metrics = []
    logger = create_and_configure_logger(str(base_path / "eval.log"))

    for fold_index in range(10):
        try:
            fold_path = base_path / f"{fold_index=}"
            # make sure the fold exists
            assert fold_path.exists(), f"Fold {fold_index}: {fold_path} does not exist!"
            cfg = get_config(config_path=fold_path)
            checkpoint_path = get_checkpoint_path(
                fold_path, checkpoint_template, logger=logger
            )
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            model_class = ModelMapping[cfg.model.model_params.model_name].value

            # Hack to load the model with the correct model_args
            model = model_class.load_from_checkpoint(  # type: ignore
                checkpoint_path=checkpoint_path,
            )
            # Recreate the config with the correct args (we still need cfg for the model class!)
            data_args = model.hparams["data_args"]
            data_path_args = model.hparams["data_path_args"]
            trainer_args = model.hparams["trainer_args"]
            model_args = model.hparams["model_args"]
            model_args.is_training = False
            cfg = Args(
                data=data_args,
                data_path=data_path_args,
                model=model_args,
                trainer=trainer_args,
            )
            # print.pprint(cfg)
            model = model_class.load_from_checkpoint(  # type: ignore
                checkpoint_path=checkpoint_path,
                model_args=model_args,
            )

            trainer = configure_trainer(args=cfg.trainer, logger=None, callbacks=[])
        except AssertionError as e:
            logger.info(f"Skipping fold {fold_index}!")
            logger.info(e)
            continue

        dm = ETDataModuleFast(cfg)

        results = trainer.predict(model, datamodule=dm)

        assert results is not None, "Results are None!"
        for index, eval_type_results in enumerate(results):
            # https://github.com/lacclab/Cognitive-State-Decoding/blob/033b8191fb7048a0e2646ff730f95f4a1ff99667/src/configs/data_args.py#L51-L52
            if index in [0, 1, 2]:
                items, subjects, test_condition = zip(
                    *[
                        (i[5], i[6], i[10])  # type: ignore
                        for i in dm.val_datasets[index].ordered_key_list
                    ]  # type: ignore
                )
            else:
                items, subjects, test_condition = zip(
                    *[
                        (i[5], i[6], i[10])  # type: ignore
                        for i in dm.test_datasets[index % 3].ordered_key_list
                    ]  # type: ignore
                )
            labels, preds = zip(*eval_type_results)
            labels = torch.cat(labels, dim=0)  # type: ignore
            preds = torch.cat(preds, dim=0)  # type: ignore
            eval_regime = regime_names[index % 3]
            eval_type = "val" if index in [0, 1, 2] else "test"

            if preds.ndim == 1:
                binary_labels = labels
                binary_preds = preds
            else:
                binary_labels = binarize_labels(labels)  # type: ignore
                binary_preds = binarize_probs(preds)  # type: ignore
            df = pd.DataFrame(
                {
                    "subjects": subjects,
                    "items": items,
                    "condition": test_condition,
                    "binary_label": binary_labels.numpy(),
                    "binary_prediction_prob": binary_preds.numpy(),
                    "label": labels.numpy(),  # type: ignore
                    "prediction_prob": preds.numpy().tolist(),  # type: ignore
                    "eval_regime": eval_regime,
                    "eval_type": eval_type,
                    "fold_index": fold_index,
                    "ia_query": cfg.data.ia_query,
                }
            )
            group_level_metrics.append(df)

    res = pd.concat(group_level_metrics)
    res["binary_prediction"] = (res["binary_prediction_prob"] > 0.5).astype(int)
    res["is_correct"] = (res["binary_label"] == res["binary_prediction"]).astype(int)
    res["level"] = res["items"].str.split("_").str[2]

    res.to_csv(base_path / "trial_level_test_results.csv")
    logger.info(
        f"Saved trial level results to {base_path / 'trial_level_test_results.csv'}"
    )


def binarize_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Binarize labels to 0 and 1. Replace 0,1 with 1 (positive class) and 2,3 with 0 (negative).

    Args:
        labels (torch.Tensor): The labels to binarize.

    Returns:
        torch.Tensor: The binarized labels.
    """
    labels = labels.clone()
    # labels = torch.where((labels == 0) , 0, labels)
    labels = torch.where((labels == 2) | (labels == 3) | (labels == 1), 1, labels)
    labels = 1 - labels  # invert the labels
    return labels


def binarize_probs(probs: torch.Tensor, n_first_cols: int = 1) -> torch.Tensor:
    """
    Binarize probs to 0 and 1. Replace the first n columns with 1 and the remaining columns with 0.
    Note the switch of columns 0 and 1!
    by taking the max of each pair.

    """
    max_first_columns = torch.max(probs[:, :n_first_cols], dim=1, keepdim=True).values
    max_last_columns = torch.max(probs[:, n_first_cols:], dim=1, keepdim=True).values

    binarized_probs = torch.cat((max_first_columns, max_last_columns), dim=1)
    binarized_probs = torch.softmax(binarized_probs, dim=1)
    binarized_probs = binarized_probs[:, 0]

    return binarized_probs


if __name__ == "__main__":
    main()
