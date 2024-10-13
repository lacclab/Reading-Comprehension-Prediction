from abc import abstractmethod
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Literal, Optional

from sklearn import metrics
from sklearn.pipeline import Pipeline
import importlib

import torch
import numpy as np

from omegaconf import MISSING
from torch.utils.data import DataLoader

from transformers import (
    RobertaConfig,
    RobertaModel,
)

import wandb
from src.configs.constants import PredMode, ItemLevelFeaturesModes
from src.configs.model_args.base_model_args_ml import BaseMLModelArgs
from src.configs.trainer_args_ml import BaseMLTrainerArgs
from src.configs.model_args.ml_model_specific_args.LogisticRegressionMLArgs import (
    LogisticRegressionMLArgs,
)
from src.configs.model_args.ml_model_specific_args.DummyClassifierMLArgs import (
    DummyClassifierMLArgs,
)
from src.configs.model_args.ml_model_specific_args.KNearestNeighborsMLArg import (
    KNearestNeighborsMLArgs,
)
from src.configs.model_args.ml_model_specific_args.SupportVectorMachineMLArgs import (
    SupportVectorMachineMLArgs,
)

from src.external_features_extractor import (
    feature_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT,
    feature_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT,
)


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


# TODO this is not finished
class BaseMLModel:
    def __init__(self, model_args: BaseMLModelArgs, trainer_args: BaseMLTrainerArgs):
        self.num_workers = trainer_args.num_workers
        self.use_eyes_only = model_args.prediction_config.use_eyes_only
        self.use_fixation_report = model_args.use_fixation_report
        self.class_names = list(model_args.prediction_config.class_names)
        self.num_classes = len(self.class_names)
        self.average: Literal["macro"] = "macro"
        self.validate_metrics: bool = False
        self.prediction_mode: PredMode = model_args.prediction_config.prediction_mode
        self.task: Literal["binary", "multiclass"] = (
            "multiclass" if self.num_classes > 2 else "binary"
        )
        self.regime_names: list[str] = [
            "new_item",
            "new_subject",
            "new_item_and_subject",
            "all",
        ]  # This order is defined in the data module!
        if model_args.model_params.class_weights is not None:
            self.class_weights = torch.Tensor(model_args.model_params.class_weights)
            print(f"Using class weights: {self.class_weights}")
        else:
            self.class_weights = None
            print("Not using class weights")

        self._init_classifier(model_args)
        self.classless_accuracies = {}
        self.balanced_class_accuracies = {}
        self.stage_count = {}

        self.ia_features = model_args.ia_features

        #### features builder ###
        self.use_item_level_features: bool = (
            model_args.model_params.item_level_features_mode
            != ItemLevelFeaturesModes.NONE
        )
        self.use_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT: bool = (
            model_args.model_params.use_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT
        )
        self.use_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT: bool = (
            model_args.model_params.use_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT
        )

        self.bert_encoder_needed = np.any(
            [
                self.use_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT,
                self.use_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT,
            ]
        )
        if self.bert_encoder_needed:
            self.features_batch_size = trainer_args.batch_size
            # note that trainer_args.devices is the number of GPUs to use
            self.feature_builder_device = (
                "cuda" if torch.cuda.is_available() and trainer_args.devices else "cpu"
            )

            ## initialize the bert encoder ##
            # Word-Sequence Encoder
            encoder_config = RobertaConfig.from_pretrained(model_args.backbone)
            encoder_config.output_hidden_states = True
            # initiate Bert with pre-trained weights
            print("keeping Bert with pre-trained weights")
            self.bert_encoder: RobertaModel = RobertaModel.from_pretrained(
                model_args.backbone, config=encoder_config
            ).to(  # type: ignore
                self.feature_builder_device  # type: ignore
            )  # type: ignore

    def _init_classifier(self, model_args) -> None:
        """
        Initialize the classifier.
        """
        # make pipeline from model_params.sklearn_pipline
        ## empty pipline
        self.classifier = Pipeline([])
        for step_name, method_import_path in model_args.model_params.sklearn_pipeline:
            module_name, class_name = method_import_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            ClassObj = getattr(module, class_name)

            self.classifier.steps.append((step_name, ClassObj()))

        # set params
        print("Setting model params")
        model_args.model_params.init_sklearn_pipeline_params()
        self.classifier.set_params(**model_args.model_params.sklearn_pipeline_params)

    def fit(self, dm) -> None:
        """
        Fit the model to the data.

        Args:
        - dm: ETDataModuleFast - Data module for the model.
        """
        print("Fitting model")
        train_dataloader = DataLoader(
            dm.train_dataset,
            batch_size=self.features_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        train_batchs: list[BatchData] = self.unpack_data(train_dataloader)
        self.shared_fit(train_batchs)  # fit the model to the data

    def predict(self, dataset) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Predict the model on the data.

        Args:
        - dm: ETDataModuleFast - Data module for the model.
        """

        dev_dataloader = DataLoader(
            dataset,
            batch_size=self.features_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        dev_batchs: list[BatchData] = self.unpack_data(dev_dataloader)
        preds_list, probs_list = self.shared_predict(
            dev_batchs
        )  # predict the model on the data
        y_true_list = []
        for dev_batch in tqdm(dev_batchs, desc="Feature extraction (pred)"):
            y_true_list.append(dev_batch.labels)

        ordered_preds_list, ordered_probs_list, ordered_y_true_list = [], [], []
        for preds, probs, y_true, batch in zip(
            preds_list, probs_list, y_true_list, dev_batchs
        ):
            # order the labels and logits
            if self.task == "multiclass":
                ordered_preds, ordered_probs = self.order_labels_logits(
                    probs, preds, batch.answer_mappings
                )
                ordered_y_true = self.order_labels(batch.labels, batch.answer_mappings)
            else:
                # copy preds, probs and y_true
                ordered_preds, ordered_probs, ordered_y_true = preds, probs, y_true
            ordered_preds_list.append(ordered_preds)
            ordered_probs_list.append(ordered_probs)
            ordered_y_true_list.append(ordered_y_true)

        preds = torch.cat(preds_list, dim=0)
        probs = torch.cat(probs_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        ordered_preds = torch.cat(ordered_preds_list, dim=0)
        ordered_probs = torch.cat(ordered_probs_list, dim=0)
        ordered_y_true = torch.cat(ordered_y_true_list, dim=0)

        return preds, probs, y_true, ordered_preds, ordered_probs, ordered_y_true

    def evaluate(self, eval_dataset, stage: str, validation_map: str) -> None:
        """
        Evaluate the model on the data.

        Args:
        - eval_dataset: ETDataset - dataset to evaluate the model on.
        - stage: str - The stage of evaluation, either 'train' or 'val' or 'test'.
        - validation_map: str - The type of validation, either 'new_item', 'new_subject', 'new_item_and_subject', or 'all'.
        """
        self.stage = stage
        self.validation_map = validation_map

        (
            self.preds,
            self.probs,
            self.y_true,
            self.ordered_preds,
            self.ordered_probs,
            self.ordered_y_true,
        ) = self.predict(
            eval_dataset
        )  # in case of binary classification, ordered_preds and ordered_probs are the same as preds and probs

        self._on_eval_end()

    def _on_eval_end(self) -> None:
        """
        Function to run at the end of evaluation.
        """
        assert self.preds is not None
        assert self.probs is not None
        assert self.y_true is not None
        assert self.ordered_preds is not None
        assert self.ordered_probs is not None
        assert self.ordered_y_true is not None

        # ############################
        # Logic goes here

        # convert tensors to numpy
        self.preds = self.preds.numpy()
        self.probs = self.probs.numpy()
        self.y_true = self.y_true.numpy()
        self.ordered_preds = self.ordered_preds.numpy()
        self.ordered_probs = self.ordered_probs.numpy()
        self.ordered_y_true = self.ordered_y_true.numpy()

        self._log_metrics()

        # classless accuracy: % of correct predictions
        # note that it does not matter if we are using the ordered or unordered predictions
        self.classless_accuracies[self.validation_map] = metrics.accuracy_score(
            y_true=self.ordered_y_true,
            y_pred=self.ordered_preds,  # type: ignore
        )
        # balanced class accuracy: % of correct predictions per class weighted by the number of samples
        self.balanced_class_accuracies[self.validation_map] = (
            metrics.balanced_accuracy_score(
                y_true=self.ordered_y_true,
                y_pred=self.ordered_preds,  # type: ignore
            )
        )
        self.stage_count[self.validation_map] = len(self.ordered_y_true)
        # ############################

        (
            self.preds,
            self.probs,
            self.y_true,
            self.ordered_preds,
            self.ordered_probs,
            self.ordered_y_true,
        ) = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def _log_metrics(self) -> None:
        """
        Log the metrics of the evaluation.
        """

        # Log Confusion Matrix
        wandb.log(
            {
                f"Confusion_Matrix (Unordered)/{self.stage}_{self.validation_map}": wandb.plot.confusion_matrix(  # type: ignore
                    preds=self.preds,  # type: ignore
                    y_true=self.y_true,  # type: ignore
                    class_names=self.class_names,
                    title=f"Confusion_Matrix (Unordered)/{self.stage}_{self.validation_map}",
                )
            }
        )
        wandb.log(
            {
                f"Confusion_Matrix (Ordered)/{self.stage}_{self.validation_map}": wandb.plot.confusion_matrix(  # type: ignore
                    preds=self.ordered_preds,  # type: ignore
                    y_true=self.ordered_y_true,  # type: ignore
                    class_names=self.class_names,
                    title=f"Confusion_Matrix (Ordered)/{self.stage}_{self.validation_map}",
                )
            }
        )
        # Log ROC curve
        wandb.log(
            {
                f"ROC_Curve (Unordered)/{self.stage}_{self.validation_map}": wandb.plot.roc_curve(  # type: ignore
                    y_true=self.y_true,  # type: ignore
                    y_probas=self.probs,  # type: ignore
                    labels=self.class_names,
                    title=f"ROC_Curve (Unordered)/{self.stage}_{self.validation_map}",
                )
            }
        )
        wandb.log(
            {
                f"ROC_Curve (Ordered)/{self.stage}_{self.validation_map}": wandb.plot.roc_curve(  # type: ignore
                    y_true=self.ordered_y_true,  # type: ignore
                    y_probas=self.ordered_probs,  # type: ignore
                    labels=self.class_names,
                    title=f"ROC_Curve (Ordered)/{self.stage}_{self.validation_map}",
                )
            }
        )

        # per class metrics  (Unordered)
        class_names_unordered = np.array(self.class_names)[np.unique(self.y_true)]  # type: ignore
        self.per_class_metrics = metrics.classification_report(
            y_true=self.y_true,  # type: ignore
            y_pred=self.preds,  # type: ignore
            target_names=class_names_unordered,
            output_dict=True,
        )
        for class_name, metrics_dict in self.per_class_metrics.items():  # type: ignore
            if isinstance(metrics_dict, dict):
                for metric_name, value in metrics_dict.items():
                    wandb.run.summary[  # type: ignore
                        f"[Unordered] {metric_name}_{class_name}/{self.stage}_{self.validation_map}"
                    ] = value
            else:
                wandb.run.summary[  # type: ignore
                    f"[Unordered] {class_name}/{self.stage}_{self.validation_map}"
                ] = metrics_dict

        # per class metrics  (Ordered)
        class_names_ordered = np.array(self.class_names)[np.unique(self.ordered_y_true)]  # type: ignore
        self.per_class_metrics = metrics.classification_report(
            y_true=self.ordered_y_true,  # type: ignore
            y_pred=self.ordered_preds,  # type: ignore
            target_names=class_names_ordered,
            output_dict=True,
        )
        for class_name, metrics_dict in self.per_class_metrics.items():  # type: ignore
            if isinstance(metrics_dict, dict):
                for metric_name, value in metrics_dict.items():
                    wandb.run.summary[  # type: ignore
                        f"[Ordered] {metric_name}_{class_name}/{self.stage}_{self.validation_map}"
                    ] = value
            else:
                wandb.run.summary[  # type: ignore
                    f"[Ordered] {class_name}/{self.stage}_{self.validation_map}"
                ] = metrics_dict

    def on_stage_end(self) -> None:
        """
        Function to run at the end of a stage.
        """
        # Log classless accuracy
        wandb.run.summary[f"classless_accuracy/{self.stage}_average"] = np.mean(list(self.classless_accuracies.values()))  # type: ignore
        wandb.run.summary[  # type: ignore
            f"classless_accuracy/{self.stage}_weighted_average"
        ] = np.average(
            list(self.classless_accuracies.values()), weights=list(self.stage_count.values())  # type: ignore
        )
        # Log balanced classless accuracy
        wandb.run.summary[f"balanced_classless_accuracy/{self.stage}_average"] = np.mean(list(self.balanced_class_accuracies.values()))  # type: ignore
        wandb.run.summary[  # type: ignore
            f"balanced_classless_accuracy/{self.stage}_weighted_average"
        ] = np.average(
            list(self.balanced_class_accuracies.values()), weights=list(self.stage_count.values())  # type: ignore
        )
        # empty caches
        self.classless_accuracies = {}  # (validation_map) -> list of accuracies
        self.balanced_class_accuracies = {}  # (validation_map) -> list of accuracies
        self.stage_count = {}  # (validation_map) -> count of samples

    def unpack_data(self, dataloader) -> list[BatchData]:
        """
        * This is an adaptation of the 'unpack_method' from base_model.py

        Unpacks the batch into the different tensors.
        Note, the order of the tensors in the batch is defined in ETDataset!

        Args:
            dataloader: DataLoader - The dataloader to unpack.

        Returns:
            BatchData: The unpacked batch.
        # TODO Consider moving to ETDataset.__getitem__
        """

        data_list: list[BatchData] = []
        for batch in dataloader:

            data = BatchData()
            if self.use_eyes_only:
                if not self.use_fixation_report:
                    data.eyes, data.labels = batch
                else:
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
                if not self.use_fixation_report:
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

            data_list.append(data)
        return data_list

    def _features_builder(self, train_batch: BatchData) -> torch.Tensor:
        """
        Concatenate features for the model from different sources.
        Returns:
        - torch.Tensor - The concatenated features.
            shape: (batch_size, d_features)
        """
        # infer batch_size from train_batch.labels
        # unsqueeze in case of single sample
        if train_batch.labels.ndim == 0:
            for key, value in train_batch.__dict__.items():
                if isinstance(value, torch.Tensor):
                    train_batch.__dict__[key] = value.unsqueeze(0)

        features_list = []
        if self.use_item_level_features:
            if train_batch.trial_level_features is not None:
                features_list.append(
                    train_batch.trial_level_features.squeeze(1).to(
                        self.feature_builder_device
                    )
                )
            else:
                raise ValueError("No trial level features found in the batch data.")
        if self.use_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT:
            features_list.append(
                feature_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT(
                    device=self.feature_builder_device,
                    bert_encoder=self.bert_encoder,
                    question_ids=train_batch.question_ids,  # type: ignore
                    question_masks=train_batch.question_masks,  # type: ignore
                    paragraph_ids=train_batch.paragraph_input_ids,  # type: ignore
                    paragraph_masks=train_batch.paragraph_input_masks,  # type: ignore
                    dwell_time_weights=train_batch.eyes[..., self.ia_features.index("IA_DWELL_TIME")],  # type: ignore
                ).to(self.feature_builder_device)
            )
        if self.use_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT:
            features_list.append(
                feature_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT(
                    device=self.feature_builder_device,
                    bert_encoder=self.bert_encoder,
                    question_ids=train_batch.question_ids,  # type: ignore
                    question_masks=train_batch.question_masks,  # type: ignore
                    paragraph_ids=train_batch.paragraph_input_ids,  # type: ignore
                    paragraph_masks=train_batch.paragraph_input_masks,  # type: ignore
                    dwell_time_weights=train_batch.eyes[..., self.ia_features.index("IA_DWELL_TIME")],  # type: ignore
                ).to(self.feature_builder_device)
            )
        assert len(features_list) > 0, "No features found for the model."
        features = torch.cat(features_list, dim=1)

        return features

    @abstractmethod
    def shared_fit(self, train_batchs: list[BatchData]) -> None:
        """
        Shared fit method for the model.
        * Fit self.classifier to the data in batch.

        Args:
        - train_batchs: list[BatchData] - list of batches to fit the model to.

        Returns:
        - None

        """
        raise NotImplementedError

    @abstractmethod
    def shared_predict(
        self, dev_batchs: list[BatchData]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict the model on the data.

        Args:
        - dev_batchs: list[BatchData] - list of batches to predict the model on.

        Returns:
        - Lists of:
            - torch.Tensor - The predictions of the model.
            - torch.Tensor - Predictions probabilities.
        """
        raise NotImplementedError

    @staticmethod
    def order_labels_logits(
        logits, labels, answer_mapping
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        * On eval set labels to the predicted labels

        Returns:
        - torch.Tensor - The ordered labels.
        - torch.Tensor - The ordered logits.
        """
        # Get the sorted indices of answer_mapping along dimension 1
        sorted_indices = answer_mapping.argsort(dim=1)
        # Use these indices to rearrange each row in logits
        ordered_logits = torch.gather(logits, 1, sorted_indices)
        ordered_label = answer_mapping[range(answer_mapping.shape[0]), labels]

        return ordered_label, ordered_logits

    @staticmethod
    def order_labels(labels, answer_mapping) -> torch.Tensor:
        # Use these indices to rearrange each row in logits
        return answer_mapping[range(answer_mapping.shape[0]), labels]

    @staticmethod
    def order_logits(logits, answer_mapping) -> torch.Tensor:
        # Get the sorted indices of answer_mapping along dimension 1
        sorted_indices = answer_mapping.argsort(dim=1)
        # Use these indices to rearrange each row in logits
        return torch.gather(logits, 1, sorted_indices)


class LogisticRegressionMLModel(BaseMLModel):
    def __init__(
        self,
        model_args: LogisticRegressionMLArgs,
        trainer_args: BaseMLTrainerArgs,
    ):
        super().__init__(model_args, trainer_args)

    def shared_fit(self, train_batchs: list[BatchData]) -> None:
        """
        Shared fit method for the model.
        * Fit self.classifier to the data in batch.

        Args:
        - train_batchs: list[BatchData] - list of batches to fit the model to.

        Returns:
        - None

        """
        features_list: list[torch.Tensor] = []
        y_true_list = []
        for train_batch in tqdm(train_batchs, desc="Feature extraction"):
            features = self._features_builder(train_batch).to("cpu")
            features_list.append(features)
            y_true_list.append(train_batch.labels)
        features = torch.cat(features_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)

        # convert to numpy
        features = features.numpy()
        y_true = y_true.numpy()
        self.classifier.fit(features, y_true)

    def shared_predict(
        self, dev_batchs: list[BatchData]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Predict the model on the data.

        Args:
        - dev_batchs: list[BatchData] - list of batches to predict the model on.

        Returns:
        - Lists of:
            - torch.Tensor - The predictions of the model.
            - torch.Tensor - Predictions probabilities.
        """
        preds_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        for dev_batch in dev_batchs:
            features = self._features_builder(dev_batch).to("cpu")
            # convert to numpy
            features = features.numpy()
            try:
                probs = torch.Tensor(self.classifier.predict_proba(features))  # type: ignore
                preds = probs.argmax(dim=1)
            except Exception as e:
                print(e)  # the model does not have a predict_proba method
                print(
                    "Assuming the model does not have a predict_proba method\nNote that ROC will be falsely calculated."
                )
                preds = torch.Tensor(self.classifier.predict(features))
                # convert to probabilities
                probs = torch.zeros((len(preds), self.num_classes))
                probs[range(len(preds)), preds] = 1

            preds_list.append(preds)
            probs_list.append(probs)

        return preds_list, probs_list


class DummyClassifierMLModel(BaseMLModel):
    def __init__(
        self,
        model_args: DummyClassifierMLArgs,
        trainer_args: BaseMLTrainerArgs,
    ):
        super().__init__(model_args, trainer_args)

    def shared_fit(self, train_batchs: list[BatchData]) -> None:
        """
        Shared fit method for the model.
        * Fit self.classifier to the data in batch.

        Args:
        - train_batchs: list[BatchData] - list of batches to fit the model to.

        Returns:
        - None

        """
        features_list = []
        y_true_list = []
        for train_batch in train_batchs:
            # generate all features as zeros (dummy classifier)
            features = torch.zeros((train_batch.labels.shape[0], 1)).to(
                "cpu"
            )  # TODO: consider different way to get batch_size instead of using the labels
            features_list.append(features)
            y_true_list.append(train_batch.labels)
        features = torch.cat(features_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)

        # convert to numpy
        features = features.numpy()
        y_true = y_true.numpy()
        self.classifier.fit(features, y_true)

    def shared_predict(
        self, dev_batchs: list[BatchData]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Predict the model on the data.

        Args:
        - dev_batchs: list[BatchData] - list of batches to predict the model on.

        Returns:
        - Lists of:
            - torch.Tensor - The predictions of the model.
            - torch.Tensor - Predictions probabilities.
        """
        preds_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        for dev_batch in dev_batchs:
            features = torch.zeros((dev_batch.labels.shape[0], 1)).to("cpu")
            # convert to numpy
            features = features.numpy()
            try:
                probs = torch.Tensor(self.classifier.predict_proba(features))  # type: ignore
                preds = probs.argmax(dim=1)
            except Exception as e:
                print(e)  # the model does not have a predict_proba method
                print(
                    "Assuming the model does not have a predict_proba method\nNote that ROC will be falsely calculated."
                )
                preds = torch.Tensor(self.classifier.predict(features))
                # convert to probabilities
                probs = torch.zeros((len(preds), self.num_classes))
                probs[range(len(preds)), preds] = 1

            preds_list.append(preds)
            probs_list.append(probs)

        return preds_list, probs_list


class KNearestNeighborsMLModel(BaseMLModel):
    def __init__(
        self,
        model_args: KNearestNeighborsMLArgs,
        trainer_args: BaseMLTrainerArgs,
    ):
        super().__init__(model_args, trainer_args)

    def shared_fit(self, train_batchs: list[BatchData]) -> None:
        """
        Shared fit method for the model.
        * Fit self.classifier to the data in batch.

        Args:
        - train_batchs: list[BatchData] - list of batches to fit the model to.

        Returns:
        - None

        """
        features_list: list[torch.Tensor] = []
        y_true_list = []
        for train_batch in tqdm(train_batchs, desc="Feature extraction"):
            features = self._features_builder(train_batch).to("cpu")
            features_list.append(features)
            y_true_list.append(train_batch.labels)
        features = torch.cat(features_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)

        # convert to numpy
        features = features.numpy()
        y_true = y_true.numpy()
        self.classifier.fit(features, y_true)

    def shared_predict(
        self, dev_batchs: list[BatchData]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Predict the model on the data.

        Args:
        - dev_batchs: list[BatchData] - list of batches to predict the model on.

        Returns:
        - Lists of:
            - torch.Tensor - The predictions of the model.
            - torch.Tensor - Predictions probabilities.
        """
        preds_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        for dev_batch in dev_batchs:
            features = self._features_builder(dev_batch).to("cpu")
            # convert to numpy
            features = features.numpy()
            try:
                probs = torch.Tensor(self.classifier.predict_proba(features))  # type: ignore
                preds = probs.argmax(dim=1)
            except Exception as e:
                print(e)  # the model does not have a predict_proba method
                print(
                    "Assuming the model does not have a predict_proba method\nNote that ROC will be falsely calculated."
                )
                preds = torch.Tensor(self.classifier.predict(features))
                # convert to probabilities
                probs = torch.zeros((len(preds), self.num_classes))
                probs[range(len(preds)), preds] = 1

            preds_list.append(preds)
            probs_list.append(probs)

        return preds_list, probs_list


class SupportVectorMachineMLModel(BaseMLModel):
    def __init__(
        self,
        model_args: SupportVectorMachineMLArgs,
        trainer_args: BaseMLTrainerArgs,
    ):
        super().__init__(model_args, trainer_args)

    def shared_fit(self, train_batchs: list[BatchData]) -> None:
        """
        Shared fit method for the model.
        * Fit self.classifier to the data in batch.

        Args:
        - train_batchs: list[BatchData] - list of batches to fit the model to.

        Returns:
        - None

        """
        features_list: list[torch.Tensor] = []
        y_true_list = []
        for train_batch in tqdm(train_batchs, desc="Feature extraction"):
            features = self._features_builder(train_batch).to("cpu")
            features_list.append(features)
            y_true_list.append(train_batch.labels)
        features = torch.cat(features_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)

        # convert to numpy
        features = features.numpy()
        y_true = y_true.numpy()
        self.classifier.fit(features, y_true)

    def shared_predict(
        self, dev_batchs: list[BatchData]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Predict the model on the data.

        Args:
        - dev_batchs: list[BatchData] - list of batches to predict the model on.

        Returns:
        - Lists of:
            - torch.Tensor - The predictions of the model.
            - torch.Tensor - Predictions probabilities.
        """
        preds_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        for dev_batch in dev_batchs:
            features = self._features_builder(dev_batch).to("cpu")
            # convert to numpy
            features = features.numpy()
            try:
                probs = torch.Tensor(self.classifier.predict_proba(features))  # type: ignore
                preds = probs.argmax(dim=1)
            except Exception as e:
                print(e)  # the model does not have a predict_proba method
                print(
                    "Assuming the model does not have a predict_proba method\nNote that ROC will be falsely calculated."
                )
                preds = torch.Tensor(self.classifier.predict(features))
                # convert to probabilities
                probs = torch.zeros((len(preds), self.num_classes))
                probs[range(len(preds)), preds] = 1

            preds_list.append(preds)
            probs_list.append(probs)

        return preds_list, probs_list
