"""base_roberta.py - Base class for MAG and RoBERTeye models.
See 1. On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines:
https://www.semanticscholar.org/reader/8b9d77d5e52a70af37451d3db3d32781b83ea054 for parameters
"""

import torch
from pytorch_metric_learning import losses
from transformers import get_linear_schedule_with_warmup

from src.configs.constants import (
    DataRepresentation,
    ModelNames,
    PredMode,
)
from src.configs.model_args.model_specific_args.MAGArgs import MAG
from src.configs.model_args.model_specific_args.PostFusionArgs import PostFusion
from src.configs.model_args.model_specific_args.RoBERTeyeArgs import Roberteye
from src.configs.trainer_args import Base
from src.models.base_model import BaseModel


class BaseMultiModalRoberta(BaseModel):
    """
    Model for Multiple Choice Question Answering and question prediction tasks.
    """

    def __init__(
        self,
        model_args: Roberteye | MAG | PostFusion,
        trainer_args: Base,
    ):
        super().__init__(model_args, trainer_args)

        self.model_args = model_args
        self.preorder = model_args.preorder
        self.warmup_proportion = trainer_args.warmup_proportion

        print(f"##### Preorder labels: {self.preorder} #####")
        if self.model_args.add_contrastive_loss:
            self.cl_alpha = 1
            self.cl_memory_size = 1024
            self.cl_temperature = 0.07
            self.contrastive_loss = losses.CrossBatchMemory(
                loss=losses.NTXentLoss(temperature=self.cl_temperature),
                embedding_size=model_args.contrastive_loss_embd_dim,
                memory_size=self.cl_memory_size,
            )
            print(f"##### Contrastive loss added with alpha: {self.cl_alpha} #####")
            print(f"##### Contrastive loss memory size: {self.cl_memory_size} #####")
            print(f"##### Contrastive loss temperature: {self.cl_temperature} #####")
            print(f"##### Contrastive loss embedding size: {model_args.text_dim} #####")
        self.save_hyperparameters()

    def forward(
        self, input_ids, attention_mask, labels, gaze_features, gaze_positions, **kwargs
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            gaze_features=gaze_features,
            gaze_positions=gaze_positions,
            **kwargs,
        )

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)

        labels = batch_data.labels
        if self.prediction_mode in (
            PredMode.CORRECT_ANSWER,
            PredMode.CHOSEN_ANSWER,
            PredMode.QUESTION_LABEL,
            PredMode.QUESTION_n_CONDITION,
        ):
            assert batch_data.answer_mappings is not None
            answer_mappings = batch_data.answer_mappings
            if self.preorder:
                labels = answer_mappings[range(answer_mappings.shape[0]), labels]

        assert batch_data.input_masks is not None
        assert batch_data.grouped_inversions is not None

        if self.model_args.use_fixation_report:
            # assert batch_data.scanpath_pads is not None
            # assert batch_data.scanpath is not None
            assert batch_data.fixation_features is not None
            # shortest_scanpath_pad = batch_data.scanpath_pads.min()
            # longest_batch_scanpath = MAX_SCANPATH_LENGTH - shortest_scanpath_pad

            gaze_positions = (
                batch_data.grouped_inversions
            )  # [..., :longest_batch_scanpath, :]
            gaze_features = (
                batch_data.fixation_features
            )  # [..., :longest_batch_scanpath, :]
            # Slice from start till longest_batch_scanpath
            # sliced_eye_input_masks = batch_data.input_masks[
            # ..., :longest_batch_scanpath
            # ]

            # # Slice from MAX_SCANPATH_LENGTH till end
            # text_input_masks = batch_data.input_masks[..., MAX_SCANPATH_LENGTH:]
            # attention_mask = torch.cat(
            # [sliced_eye_input_masks, text_input_masks], dim=-1
            # )
            attention_mask = batch_data.input_masks

        else:
            assert batch_data.eyes is not None
            assert batch_data.input_ids is not None
            gaze_features = batch_data.eyes
            gaze_positions = batch_data.grouped_inversions
            attention_mask = batch_data.input_masks

            # # Shorten the pads
            # nonzero_indices = torch.nonzero(attention_mask)
            # max_index = torch.max(nonzero_indices[:, -1])
            # pad_len = attention_mask.shape[-1] - max_index
            # nonzero_indices = torch.nonzero(attention_mask)

            # gaze_features = gaze_features[..., :max_index]
            # gaze_positions = gaze_positions[..., :max_index]
            # attention_mask = attention_mask[..., :max_index]

        if (not self.model_args.model_params.prepend_eye_data) and (
            self.model_args.model_params.model_name == ModelNames.ROBERTEYE_MODEL
        ):  # TODO Write why this is necessary
            gaze_features = None
            gaze_positions = None

        output = self(
            input_ids=batch_data.input_ids,  # [..., :-pad_len.item()],
            attention_mask=attention_mask,
            labels=labels,
            gaze_features=gaze_features,
            gaze_positions=gaze_positions,
            output_hidden_states=True,
        )

        logits = output.logits

        if (
            self.prediction_mode
            in (
                PredMode.CHOSEN_ANSWER,
                PredMode.QUESTION_LABEL,
                PredMode.QUESTION_n_CONDITION,
            )
            and not self.preorder
        ):
            assert batch_data.answer_mappings is not None
            ordered_labels, ordered_logits = self.order_labels_logits(
                logits=logits, labels=labels, answer_mapping=batch_data.answer_mappings
            )
        else:
            ordered_labels = labels
            ordered_logits = logits

        if self.class_weights is not None:
            loss = self.calculate_weighted_loss(
                logits=logits, labels=labels, ordered_labels=ordered_labels
            )
        else:
            loss = output.loss

        if self.model_args.add_contrastive_loss:
            if (
                self.model_args.model_params.concat_or_duplicate
                == DataRepresentation.DUPLICATE
            ):
                pred_indices = [
                    label + self.num_classes * i
                    for i, label in enumerate(torch.argmax(logits, dim=1))
                ]
                last_hidden = output.hidden_states[-1][pred_indices, 0, :]
            else:
                last_hidden = output.hidden_states[-1][:, 0, :]
            contrastive_loss = self.contrastive_loss(last_hidden, ordered_labels)
            loss += self.cl_alpha * contrastive_loss
        return ordered_labels, loss, ordered_logits, labels, logits

    def configure_optimizers(self):
        # Define the optimizer
        assert self.warmup_proportion is not None
        stepping_batches = self.trainer.estimated_stepping_batches

        # Copied from bert
        param_optimizer = list(self.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(stepping_batches * self.warmup_proportion),
            num_training_steps=stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
