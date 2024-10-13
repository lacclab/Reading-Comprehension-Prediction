""" This module contains the base model class and the multimodal model class. """ ""
from functools import partial
import itertools
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import lightning.pytorch as pl
import matplotlib.lines as matplotlib_lines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import torchmetrics.classification as cls_metrics
import torchmetrics.wrappers as metric_wrappers
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import MISSING
from torch import nn

import wandb
from src.configs.constants import PredMode
from src.configs.model_args.base_model_args import BaseModelArgs
from src.configs.trainer_args import Base


@dataclass
class BatchData:
    """
    dataclass for batch items.
    PARAGRAPH_INPUT_IDS - input_ids of each tokenized paragraph (batch_size, MAX_SEQ_LENGTH)
    EYES - et_data_enriched feature vectors for each IA_ID (batch_size, No. IA_IDs, d_IA_features)
    SCANPATH - A series of IA_IDs for each trial scanpath (batch_size,MAX_SCANPATH_LENGTH)
    FIXATIONS_FEATURES - fixations_enriched feature vectors for each fixation with
                            IA features from et_data_enriched (batch_size, MAX_SCANPATH_LENGTH,
                                                                d_IA_features + d_fixation_features)
    INVERSIONS - For each paragraph a list, where the value at entry i is the IA_ID that is
                    associated with the i-th token in the paragraph (batch_size, MAX_SEQ_LENGTH)

    Attributes:
    - eyes: Optional[torch.Tensor] - et_data_enriched feature vectors for each IA_ID.
    - fixation_features: Optional[torch.Tensor] - fixations feature vectors for each fixation.
    - labels: Optional[torch.Tensor] - Labels associated with the batch.
    - fixation_pads: Optional[torch.Tensor] - Feature vectors for fixations.
    - scanpath: Optional[torch.Tensor] - A series of IA_IDs for each trial scanpath.
    - scanpath_pads: Optional[torch.Tensor] - Feature vectors for IA_IDs in scanpath.
    - paragraph_input_ids: Optional[torch.Tensor] - Input_ids of each tokenized paragraph.
    - paragraph_input_masks: Optional[torch.Tensor] - Masks for paragraph input_ids.
    - input_ids: Optional[torch.Tensor] - Input_ids of tokenized paragraphs.
    - input_masks: Optional[torch.Tensor] - Masks for input_ids.
    - answer_mappings: Optional[torch.Tensor] - Mappings associated with answers.
    - inversions: Optional[torch.Tensor] - For each paragraph, IA_IDs associated with tokens.
    - inversions_pads: Optional[torch.Tensor] - Pads for IA_IDs in inversions.
    """

    eyes: Optional[torch.Tensor] = field(default=None)
    fixation_features: Optional[torch.Tensor] = field(default=None)
    labels: torch.Tensor = MISSING
    fixation_pads: Optional[torch.Tensor] = field(default=None)
    scanpath: Optional[torch.Tensor] = field(default=None)
    scanpath_pads: Optional[torch.Tensor] = field(default=None)
    paragraph_input_ids: Optional[torch.Tensor] = field(default=None)
    paragraph_input_masks: Optional[torch.Tensor] = field(default=None)
    input_ids: Optional[torch.Tensor] = field(default=None)
    input_masks: Optional[torch.Tensor] = field(default=None)
    answer_mappings: Optional[torch.Tensor] = field(default=None)
    inversions: Optional[torch.Tensor] = field(default=None)
    inversions_pads: Optional[torch.Tensor] = field(default=None)
    grouped_inversions: Optional[torch.Tensor] = field(default=None)
    trial_level_features: Optional[torch.Tensor] = field(default=None)
    question_ids: Optional[torch.Tensor] = field(default=None)
    question_masks: Optional[torch.Tensor] = field(default=None)


class BaseModel(pl.LightningModule):
    """Base model class for the multi-modal models."""

    def __init__(self, model_args: BaseModelArgs, trainer_args: Base):
        super().__init__()
        self.use_eyes_only = model_args.prediction_config.use_eyes_only
        self.add_answers = model_args.prediction_config.add_answers
        self.use_fixation_report = model_args.use_fixation_report
        self.learning_rate = trainer_args.learning_rate
        self.weight_decay = trainer_args.weight_decay
        self.batch_size = model_args.batch_size
        self.class_names = list(model_args.prediction_config.class_names)
        self.num_classes = len(self.class_names)
        self.average: Literal["macro"] = "macro"
        self.validate_metrics: bool = False
        self.prediction_mode: PredMode = model_args.prediction_config.prediction_mode
        self.task: Literal["binary", "multiclass"] = (
            "multiclass" if self.num_classes > 2 else "binary"
        )
        self.regime_names: list = [
            "new_item",
            "new_subject",
            "new_item_and_subject",
            # "all", #! Not supported any more. Requires adding dataloaders and metrics.
        ]  # This order is defined in the data module!

        self.val_max_acc_dict = {
            k: {m: 0.0 for m in ["average", "weighted_average"]}
            for k in ["Balanced", "Classless"]
        }

        if model_args.model_params.class_weights is not None:
            self.class_weights = torch.Tensor(model_args.model_params.class_weights)
            print(f"Using class weights: {self.class_weights}")
        else:
            self.class_weights = None
            print("Not using class weights")

        (
            metrics,
            confusion_matrix,
            roc,
            classless_accuracy,
            balanced_accuracy,
        ) = self.configure_metrics()

        self.train_metrics = metrics.clone(postfix="/train")
        self.val_metrics_list = nn.ModuleList(
            [
                metrics.clone(postfix=f"/val_{regime_name}")
                for regime_name in self.regime_names
            ]
        )
        self.test_metrics_list = nn.ModuleList(
            [
                metrics.clone(postfix=f"/test_{regime_name}")
                for regime_name in self.regime_names
            ]
        )

        self.train_cm = confusion_matrix.clone()
        self.val_cm_list = nn.ModuleList(
            [confusion_matrix.clone() for _ in self.regime_names]
        )
        self.test_cm_list = nn.ModuleList(
            [confusion_matrix.clone() for _ in self.regime_names]
        )
        self.train_roc = roc.clone()
        self.val_roc_list = nn.ModuleList([roc.clone() for _ in self.regime_names])
        self.test_roc_list = nn.ModuleList([roc.clone() for _ in self.regime_names])

        self.train_classless_accuracy = classless_accuracy.clone()
        self.val_classless_accuracy_list = nn.ModuleList(
            [classless_accuracy.clone() for _ in self.regime_names]
        )
        self.test_classless_accuracy_list = nn.ModuleList(
            [classless_accuracy.clone() for _ in self.regime_names]
        )

        self.train_balanced_accuracy = balanced_accuracy.clone()
        self.val_balanced_accuracy_list = nn.ModuleList(
            [balanced_accuracy.clone() for _ in self.regime_names]
        )
        self.test_balanced_accuracy_list = nn.ModuleList(
            [balanced_accuracy.clone() for _ in self.regime_names]
        )

        if self.num_classes > 2:
            binary_roc = cls_metrics.BinaryROC(
                validate_args=self.validate_metrics,
            )
            self.binary_train_roc = binary_roc.clone()
            self.binary_val_roc_list = nn.ModuleList(
                [binary_roc.clone() for _ in self.regime_names]
            )
            self.binary_test_roc_list = nn.ModuleList(
                [binary_roc.clone() for _ in self.regime_names]
            )

        self.loss = nn.CrossEntropyLoss()

        print(f"##### Using {self.task=} metrics #####")
        print(f"##### {self.prediction_mode=} prediction mode #####")
        print(f"##### Using classes {self.class_names} #####")
        print(f"##### {self.regime_names=} #####")
        print(f"##### {self.use_eyes_only=} #####")
        print(f"##### {self.use_fixation_report=} #####")
        print(f"##### {self.weight_decay=} #####")
        print(f"##### {self.learning_rate=} #####")

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError

    def unpack_batch(self, batch: list) -> BatchData:
        """
        Unpacks the batch into the different tensors.
        Note, the order of the tensors in the batch is defined in ETDataset!

        Args:
            batch (list): The batch to unpack.

        Returns:
            BatchData: The unpacked batch.
        # TODO Consider moving to ETDataset.__getitem__
        """
        data = BatchData()
        if self.use_eyes_only:
            if self.use_fixation_report:
                (
                    data.eyes,
                    data.labels,
                    data.fixation_features,
                    data.fixation_pads,
                    data.scanpath,
                    data.scanpath_pads,
                    data.trial_level_features,
                ) = batch

            else:
                data.eyes, data.labels = batch
        elif not self.use_fixation_report:
            (
                data.input_ids,
                data.input_masks,
                data.labels,
                data.eyes,
                data.answer_mappings,
                data.grouped_inversions,
                data.question_ids,
                data.question_masks,
                data.paragraph_input_ids,
                data.paragraph_input_masks,
            ) = batch
        else:
            (
                data.paragraph_input_ids,
                data.paragraph_input_masks,
                data.input_ids,
                data.input_masks,
                data.labels,
                data.eyes,
                data.answer_mappings,
                data.fixation_features,
                data.fixation_pads,
                data.scanpath,
                data.scanpath_pads,
                data.inversions,
                data.inversions_pads,
                data.grouped_inversions,
                data.trial_level_features,
            ) = batch
        return data

    @abstractmethod
    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def process_step(
        self, batch: list, step_type: str, dataloader_idx=0
    ) -> torch.Tensor:
        labels, loss, logits, unordered_labels, unordered_logits = self.shared_step(
            batch
        )

        classless_predictions = (
            torch.argmax(unordered_logits, dim=1) == unordered_labels
        )
        classless_label = torch.ones_like(classless_predictions)

        metrics = self.get_metrics_map(step_type, dataloader_idx)

        probs = logits.softmax(dim=1)
        metrics["balanced_accuracy"](
            probs.argmax(dim=1), labels
        )  # * must be one entry per class and sample, or after argmax.

        if self.num_classes == 2:
            probs = probs[:, 1]

        metrics["classless_accuracy"](classless_predictions, classless_label)
        metrics["metrics"].update(probs, labels)
        metrics["cm"].update(probs, labels)
        metrics["roc"].update(probs, labels)
        if self.num_classes > 2:
            binary_probs = self.binarize_probs(logits)
            binary_labels = self.binarize_labels(labels)
            metrics["binary_roc"].update(binary_probs, binary_labels)

        return loss

    def log_loss(self, loss: torch.Tensor, step_type: str, dataloader_idx=0) -> None:
        if step_type == "train":
            name = "loss/train"
        else:
            name = f"loss/{step_type}_{self.regime_names[dataloader_idx]}"

        self.log(
            name=name,
            value=loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def training_step(self, batch: list, _) -> torch.Tensor:
        loss = self.process_step(batch, "train")
        self.log_loss(loss, "train")
        return loss

    def validation_step(self, batch: list, _, dataloader_idx=0) -> torch.Tensor:
        loss = self.process_step(
            batch=batch, step_type="val", dataloader_idx=dataloader_idx
        )
        self.log_loss(loss, "val", dataloader_idx)
        return loss

    def test_step(self, batch: list, _, dataloader_idx=0) -> torch.Tensor:
        loss = self.process_step(
            batch=batch, step_type="test", dataloader_idx=dataloader_idx
        )
        self.log_loss(loss=loss, step_type="test", dataloader_idx=dataloader_idx)
        return loss

    def predict_step(
        self,
        batch: list,
        _,
        dataloader_idx=0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            labels,
            unused_loss,
            logits,
            unused_unordered_labels,
            unused_unordered_logits,
        ) = self.shared_step(batch)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probs = logits.softmax(dim=1)

        if self.num_classes == 2:
            probs = probs[:, 1]
        return labels, probs

    def process_epoch_end(
        self, step_type: str, regime_name: str, index=0
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        metrics = self.get_metrics_map(step_type=step_type, index=index)

        cm_torch = metrics["cm"].compute()
        cm_formatted = self.format_confusion_matrix(cm=cm_torch)
        self.log_confusion_matrix(
            cm_data=cm_formatted, title=f"ConfusionMatrix/{step_type}_{regime_name}"
        )

        metrics["roc"].compute()
        self.log_roc(roc=metrics["roc"], title=f"ROC/{step_type}_{regime_name}")

        if self.num_classes > 2:
            metrics["binary_roc"].compute()
            self.log_roc(
                roc=metrics["binary_roc"], title=f"BinaryROC/{step_type}_{regime_name}"
            )
            metrics["binary_roc"].reset()

        computed_metrics = metrics["metrics"].compute()
        computed_classless_accuracy = metrics["classless_accuracy"].compute()
        computed_balanced_accuracy = metrics["balanced_accuracy"].compute()
        all_metrics = computed_metrics | {
            f"Classless_Accuracy/{step_type}_{regime_name}": computed_classless_accuracy,
            f"Balanced_Accuracy/{step_type}_{regime_name}": computed_balanced_accuracy,
        }

        self.log_dict(
            dictionary=all_metrics,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        metrics["roc"].reset()
        metrics["metrics"].reset()
        metrics["cm"].reset()
        metrics["classless_accuracy"].reset()
        metrics["balanced_accuracy"].reset()

        return (
            computed_classless_accuracy,
            int(cm_torch.sum().item()),
            computed_balanced_accuracy,
        )  # ::int shouldn't be necessary

    def on_train_epoch_end(self) -> None:
        name_ = "train"
        self.process_epoch_end(step_type=name_, regime_name=name_)

    def on_validation_epoch_end(self) -> None:
        classless_accuracy_values, counts, balanced_accuracy_values = zip(
            *[
                self.process_epoch_end(
                    step_type="val", index=val_index, regime_name=regime_name
                )
                for val_index, regime_name in enumerate(self.regime_names)
            ]
        )

        balanced_accuracy_values = [x.item() for x in balanced_accuracy_values]
        classless_accuracy_values = [x.item() for x in classless_accuracy_values]

        for k in ["Balanced", "Classless"]:
            accuracy_values = (
                balanced_accuracy_values
                if k == "Balanced"
                else classless_accuracy_values
            )
            for m in ["average", "weighted_average"]:
                # calculate the current value, update the current and maximum values
                curr_value = (
                    np.average(accuracy_values)
                    if m == "average"
                    else np.average(accuracy_values, weights=counts)
                )
                assert isinstance(curr_value, float)
                self.val_max_acc_dict[k][m] = max(
                    [self.val_max_acc_dict[k][m], curr_value]
                )

                partial_acc_log = partial(
                    self.log,
                    prog_bar=False,
                    on_epoch=True,
                    on_step=False,
                    add_dataloader_idx=False,
                    sync_dist=True,
                )

                partial_acc_log(
                    name=f"{k}_Accuracy/val_{m}",
                    value=curr_value,
                )
                partial_acc_log(
                    name=f"{k}_Accuracy/val_best_epoch_{m}",
                    value=self.val_max_acc_dict[k][m],
                )

    def on_test_epoch_end(self) -> None:
        for test_index, regime_name in enumerate(self.regime_names):
            self.process_epoch_end(
                step_type="test", index=test_index, regime_name=regime_name
            )

    def log_roc(
        self, roc: cls_metrics.MulticlassROC | cls_metrics.BinaryROC, title: str
    ) -> None:
        ax_: plt.axes.Axes
        fig_: Any
        fig_, ax_ = roc.plot(score=True)
        ax_.set_title(title)
        ax_.set_xlabel("False Positive Rate")
        ax_.set_ylabel("True Positive Rate")

        # Add a straight line from (0,0) to (1,1) with the legend "Random"
        line = matplotlib_lines.Line2D(
            [0, 1], [0, 1], color="red", linestyle="--", label="Random"
        )
        ax_.add_line(line)
        ax_.legend()

        self.logger.experiment.log(  # type: ignore
            {
                title: wandb.Image(fig_, caption=title),
            }
        )
        # close the figure to prevent memory leaks
        plt.close(fig_)

    def configure_optimizers(self):
        # Define the optimizer
        if self.weight_decay:
            return torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def format_confusion_matrix(self, cm: torch.Tensor) -> list[list]:
        """
        Formats a confusion matrix into a list of lists containing
        class names and their corresponding values.

        Args:
            cm (torch.Tensor): The confusion matrix to format.

        Returns:
            list[list]: A list of lists containing class names and their corresponding values.
        """
        class_names = self.class_names
        cm_list = cm.tolist()
        return [
            [class_names[i], class_names[j], cm_list[i][j]]
            for i, j in itertools.product(
                range(self.num_classes), range(self.num_classes)
            )
        ]

    def log_confusion_matrix(self, cm_data: list[list], title: str) -> None:
        """
        Logs a confusion matrix to the experiment logger.

        Args:
            cm_data (list[list]): A list of lists representing the confusion matrix.
                            Each inner list should contain the actual class name,
                            the predicted and the number of values.
            title (str): The title to use for the confusion matrix plot.

        Returns:
            None
        """
        if isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger.experiment
            fields = {
                "Actual": "Actual",
                "Predicted": "Predicted",
                "nPredictions": "nPredictions",
            }

            wandb_logger.log(
                {
                    title: wandb_logger.plot_table(
                        "lacc-lab/multi-run-confusion-matrix",
                        wandb.Table(
                            columns=["Actual", "Predicted", "nPredictions"],
                            data=cm_data,
                        ),
                        fields,
                        {"title": title},
                    ),
                }
            )
        else:
            warnings.warn("No wandb logger found, cannot log confusion matrix")

    def binarize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Binarize labels to 0 and 1. Replace 0,1 with 1 (positive class) and 2,3 with 0 (negative).

        Args:
            labels (torch.Tensor): The labels to binarize.

        Returns:
            torch.Tensor: The binarized labels.
        """
        labels = labels.clone()
        labels = torch.where((labels == 0) | (labels == 1), 1, labels)
        labels = torch.where((labels == 2) | (labels == 3), 0, labels)
        return labels.to(self.device)

    def binarize_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Binarize probs to 0 and 1. Replace columns 0,1 with 1 and 2,3 with 0.
        Note the switch of columns 0 and 1!
        by taking the max of each pair.

        """
        max_first_two_cols = torch.max(probs[:, :2], dim=1, keepdim=True).values
        max_last_two_cols = torch.max(probs[:, 2:], dim=1, keepdim=True).values

        binarized_probs = torch.cat((max_first_two_cols, max_last_two_cols), dim=1)
        binarized_probs = torch.softmax(binarized_probs, dim=1)
        binarized_probs = binarized_probs[:, 0]

        return binarized_probs.to(self.device)

    def configure_metrics(self):
        """Configures the metrics for the model."""
        if self.task == "multiclass":
            print("Using multiclass metrics")
            metrics = torchmetrics.MetricCollection(
                {
                    "AUROC": cls_metrics.MulticlassAUROC(
                        num_classes=self.num_classes,
                        average=self.average,
                        validate_args=self.validate_metrics,
                        # thresholds=10,
                    ),
                    "Accuracy": cls_metrics.MulticlassAccuracy(
                        num_classes=self.num_classes,
                        average=self.average,
                        validate_args=self.validate_metrics,
                    ),
                    "F1Score": cls_metrics.MulticlassF1Score(
                        num_classes=self.num_classes,
                        average=self.average,
                        validate_args=self.validate_metrics,
                    ),
                    "Precision": cls_metrics.MulticlassPrecision(
                        num_classes=self.num_classes,
                        average=self.average,
                        validate_args=self.validate_metrics,
                    ),
                    "Recall": cls_metrics.MulticlassRecall(
                        num_classes=self.num_classes,
                        average=self.average,
                        validate_args=self.validate_metrics,
                    ),
                    "MulticlassAccuracy": metric_wrappers.ClasswiseWrapper(
                        cls_metrics.MulticlassAccuracy(
                            num_classes=self.num_classes,
                            average=None,
                            validate_args=self.validate_metrics,
                        ),
                        labels=self.class_names,
                    ),
                    "MulticlassAUROC": metric_wrappers.ClasswiseWrapper(
                        cls_metrics.MulticlassAUROC(
                            num_classes=self.num_classes,
                            average=None,
                            validate_args=self.validate_metrics,
                        ),
                        labels=self.class_names,
                    ),
                    "MulticlassF1Score": metric_wrappers.ClasswiseWrapper(
                        cls_metrics.MulticlassF1Score(
                            num_classes=self.num_classes,
                            average=None,
                            validate_args=self.validate_metrics,
                        ),
                        labels=self.class_names,
                    ),
                    "MulticlassPrecision": metric_wrappers.ClasswiseWrapper(
                        cls_metrics.MulticlassPrecision(
                            num_classes=self.num_classes,
                            average=None,
                            validate_args=self.validate_metrics,
                        ),
                        labels=self.class_names,
                    ),
                    "MulticlassRecall": metric_wrappers.ClasswiseWrapper(
                        cls_metrics.MulticlassRecall(
                            num_classes=self.num_classes,
                            average=None,
                            validate_args=self.validate_metrics,
                        ),
                        labels=self.class_names,
                    ),
                }
            )

            confusion_matrix = cls_metrics.MulticlassConfusionMatrix(
                num_classes=self.num_classes, validate_args=self.validate_metrics
            )

            roc = cls_metrics.MulticlassROC(
                num_classes=self.num_classes,
                validate_args=self.validate_metrics,
            )

        elif self.task == "binary":
            print("Using binary metrics")
            metrics = torchmetrics.MetricCollection(
                {
                    "AUROC": cls_metrics.BinaryAUROC(
                        validate_args=self.validate_metrics,
                        # thresholds=10,
                    ),
                    "F1Score": cls_metrics.BinaryF1Score(
                        validate_args=self.validate_metrics,
                    ),
                    "Precision": cls_metrics.BinaryPrecision(
                        validate_args=self.validate_metrics,
                    ),
                    "Recall": cls_metrics.BinaryRecall(
                        validate_args=self.validate_metrics,
                    ),
                    "Accuracy": cls_metrics.BinaryAccuracy(
                        validate_args=self.validate_metrics,
                    ),
                }
            )

            confusion_matrix = cls_metrics.BinaryConfusionMatrix(
                validate_args=self.validate_metrics
            )

            roc = cls_metrics.BinaryROC(
                validate_args=self.validate_metrics,
            )

        else:
            raise ValueError(f"Unknown task: {self.task}")

        balanced_accuracy = cls_metrics.MulticlassAccuracy(
            num_classes=self.num_classes,
            average=self.average,
            validate_args=self.validate_metrics,
        )  # TODO Currently separate because expects preds or probs for each class which is not the case in binary case. Can probably send to metrics different values than the rest, check it out.
        classless_accuracy = cls_metrics.BinaryAccuracy(
            validate_args=self.validate_metrics,
        )  # TODO put in each of the tasks instead of here, and use the balanced accuracy framework similarly. Cleanup afterwards.

        return metrics, confusion_matrix, roc, classless_accuracy, balanced_accuracy

    def get_metrics_map(self, step_type: str, index=0) -> dict[str, Any]:
        metrics_map = {
            "train": {
                "cm": self.train_cm,
                "roc": self.train_roc,
                "metrics": self.train_metrics,
                "classless_accuracy": self.train_classless_accuracy,
                "balanced_accuracy": self.train_balanced_accuracy,
            },
            "val": {
                "cm": self.val_cm_list[index],
                "roc": self.val_roc_list[index],
                "metrics": self.val_metrics_list[index],
                "classless_accuracy": self.val_classless_accuracy_list[index],
                "balanced_accuracy": self.val_balanced_accuracy_list[index],
            },
            "test": {
                "cm": self.test_cm_list[index],
                "roc": self.test_roc_list[index],
                "metrics": self.test_metrics_list[index],
                "classless_accuracy": self.test_classless_accuracy_list[index],
                "balanced_accuracy": self.test_balanced_accuracy_list[index],
            },
        }

        if self.num_classes > 2:
            metrics_map["train"]["binary_roc"] = self.binary_train_roc
            metrics_map["val"]["binary_roc"] = self.binary_val_roc_list[index]
            metrics_map["test"]["binary_roc"] = self.binary_test_roc_list[index]

        return metrics_map[step_type]

    def calculate_weighted_loss(self, logits, labels, ordered_labels):
        """
        Calculates the weighted loss for a batch of predictions.

        Parameters:
        logits (torch.Tensor): The predicted values from the model. Shape: (batch_size, num_classes)
        labels (torch.Tensor): The true labels for the batch. Shape: (batch_size,)
        ordered_labels (torch.Tensor): The labels ordered by their frequency. Shape: (batch_size,)

        Returns:
        torch.Tensor: The weighted loss for the batch.

        Raises:
        AssertionError: If class weights are not provided.

        Notes:
        The loss is calculated as the sum of the individual losses for each sample,
        where each individual loss is the cross-entropy loss for that sample multiplied by its weight.
        The weight for a sample is determined by its true label.
        """
        assert self.class_weights is not None, "Class weights must be defined"
        batch_size = logits.shape[0]
        base_weights = self.class_weights.repeat(batch_size, 1).to(ordered_labels)

        samples_weight = torch.gather(base_weights, 1, ordered_labels.unsqueeze(0))
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits, labels)
        weighted_losses = losses * samples_weight
        return weighted_losses.sum() / samples_weight.sum()

    @staticmethod
    def order_labels_logits(
        logits, labels, answer_mapping
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the sorted indices of answer_mapping along dimension 1
        sorted_indices = answer_mapping.argsort(dim=1)
        # Use these indices to rearrange each row in logits
        ordered_logits = torch.gather(logits, 1, sorted_indices)
        ordered_label = answer_mapping[range(answer_mapping.shape[0]), labels]

        return ordered_label, ordered_logits
