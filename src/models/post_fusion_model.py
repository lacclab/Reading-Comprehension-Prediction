import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaModel

from src.configs.constants import MAX_SCANPATH_LENGTH, PredMode
from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.model_specific_args.PostFusionArgs import PostFusion
from src.models.base_roberta import BaseMultiModalRoberta


class PostFusionModel(BaseMultiModalRoberta):
    def __init__(
        self,
        model_args: PostFusion,
        trainer_args,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ) -> None:
        super().__init__(model_args=model_args, trainer_args=trainer_args)

        self.preorder = model_args.preorder
        print(f"preorder: {self.preorder}")
        self.d_eyes = model_args.fixation_dim
        self.d_conv = model_args.text_dim // 2
        self.sep_token_id = self.model_args.sep_token_id

        self.use_attn_mask = model_args.model_params.use_attn_mask

        self.fixation_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.d_eyes,
                out_channels=self.d_conv // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm1d(num_features=self.d_conv // 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=self.d_conv // 2,
                out_channels=self.d_conv,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm1d(num_features=self.d_conv),
            torch.nn.ReLU(),
        )

        self.roberta = RobertaModel.from_pretrained(
            pretrained_model_name_or_path=model_args.backbone
        )

        if model_args.freeze:
            # freeze the roberta model
            for param in self.roberta.parameters():  # type: ignore
                param.requires_grad = False

        self.cross_att_eyes_p = torch.nn.MultiheadAttention(
            embed_dim=self.d_conv,
            num_heads=1,
            dropout=model_args.model_params.cross_attention_dropout,
            kdim=model_args.text_dim,
            vdim=model_args.text_dim,
            batch_first=True,
        )

        self.cross_att_agg_eyes_q = torch.nn.MultiheadAttention(
            embed_dim=model_args.text_dim,
            num_heads=1,
            dropout=model_args.model_params.cross_attention_dropout,
            kdim=model_args.text_dim,
            vdim=model_args.text_dim,
            batch_first=True,
        )

        self.project_to_text_dim = torch.nn.Sequential(
            torch.nn.Dropout(p=model_args.model_params.eye_projection_dropout),
            torch.nn.Linear(
                in_features=model_args.text_dim,
                out_features=model_args.text_dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=model_args.model_params.eye_projection_dropout),
            torch.nn.Linear(
                in_features=model_args.text_dim,
                out_features=model_args.text_dim,
            ),
            torch.nn.LeakyReLU(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=model_args.text_dim, out_features=model_args.text_dim // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=model_args.text_dim // 2, out_features=self.num_classes
            ),
        )

        if model_args.freeze:
            # Freeze all model parameters except specific ones
            for name, param in self.named_parameters():
                if (
                    name.startswith("cross_att_eyes_p")
                    or name.startswith("cross_att_agg_eyes_q")
                    or name.startswith("project_to_text_dim")
                    or name.startswith("classifier")
                    or name.startswith("fixation_encoder")
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.save_hyperparameters()

    def shared_step(
        self, batch: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)

        labels = batch_data.labels
        if self.prediction_mode in (
            PredMode.CORRECT_ANSWER,
            PredMode.CHOSEN_ANSWER,
            PredMode.QUESTION_LABEL,
        ):
            assert batch_data.answer_mappings is not None
            answer_mappings = batch_data.answer_mappings
            if self.preorder:
                # * Order the labels according to the answer mappings
                labels = answer_mappings[range(answer_mappings.shape[0]), labels]

        if self.model_args.use_fixation_report:
            assert batch_data.scanpath_pads is not None
            assert batch_data.scanpath is not None
            assert batch_data.fixation_features is not None
            assert batch_data.input_masks is not None
            assert batch_data.grouped_inversions is not None

            shortest_scanpath_pad = batch_data.scanpath_pads.min()
            longest_batch_scanpath = MAX_SCANPATH_LENGTH - shortest_scanpath_pad

            use_attn_mask = self.use_attn_mask
            if use_attn_mask:
                eye_text_attn_mask = batch_data.grouped_inversions[
                    ..., :longest_batch_scanpath, :  # TODO okay? Others not sliced?
                ]
            else:
                eye_text_attn_mask = None
            gaze_features = batch_data.fixation_features[
                ..., :longest_batch_scanpath, :
            ]
            # Slice from start till longest_batch_scanpath
            # unused_sliced_eye_input_masks = batch_data.input_masks[
            #     ..., :longest_batch_scanpath
            # ]

            # # Slice from MAX_SCANPATH_LENGTH till end
            # text_input_masks = batch_data.input_masks[..., MAX_SCANPATH_LENGTH:]
            # attention_mask = torch.cat(
            #     [sliced_eye_input_masks, text_input_masks], dim=-1
            # )

        else:
            assert batch_data.eyes is not None
            assert batch_data.input_ids is not None
            gaze_features = batch_data.eyes
            # gaze_positions = batch_data.grouped_inversions
            # unused_attention_mask = batch_data.input_masks

        # permute the gaze_features to (batch_size, d_eyes, max_eye_len)
        gaze_features = gaze_features.permute(0, 2, 1)

        logits, x = self(
            input_ids=batch_data.input_ids,
            attention_mask=batch_data.input_masks,
            gaze_features=gaze_features,
            eye_text_attn_mask=eye_text_attn_mask,
        )

        if (
            self.prediction_mode in (PredMode.CHOSEN_ANSWER, PredMode.QUESTION_LABEL)
            and not self.preorder
        ):
            # * If we didn't reorder the labels in the beginning, we do it here
            assert batch_data.answer_mappings is not None
            squeezed_logits = logits.squeeze()
            if squeezed_logits.ndim == 1:
                squeezed_logits = squeezed_logits.unsqueeze(0)
            ordered_labels, ordered_logits = self.order_labels_logits(
                logits=squeezed_logits,
                labels=labels,
                answer_mapping=batch_data.answer_mappings,
            )
        else:
            ordered_labels = labels
            ordered_logits = logits

        if self.class_weights is not None:
            loss = self.calculate_weighted_loss(
                logits=logits, labels=labels, ordered_labels=ordered_labels
            )
        else:
            squeezed_logits = logits.squeeze()
            if squeezed_logits.ndim == 1:
                squeezed_logits = squeezed_logits.unsqueeze(0)
            loss = self.loss(squeezed_logits, labels)

        if self.model_args.add_contrastive_loss:
            if x.ndim == 1:
                x = x.unsqueeze(0)
            contrastive_loss = self.contrastive_loss(x, ordered_labels)
            loss += self.cl_alpha * contrastive_loss
        return ordered_labels, loss, ordered_logits.squeeze(), labels, logits.squeeze()

    def forward(  # type: ignore #TODO fix type hinting
        self,
        input_ids,
        attention_mask,
        gaze_features,
        eye_text_attn_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_fixations = self.fixation_encoder(
            gaze_features
        )  # (batch_size, d_conv, max_eye_len)
        encoded_fixations = encoded_fixations.permute(
            0, 2, 1
        )  # (batch_size, max_eye_len, d_conv)

        encoded_word_seq = self.roberta(
            input_ids, attention_mask
        ).last_hidden_state  # (batch_size, max_seq_len, text_dim) # type: ignore

        p_embds, q_embds, a_embeds = self.split_context_embds_batched(
            encoded_word_seq, input_ids
        )

        eye_text_attn = self.cross_att_eyes_p(
            query=encoded_fixations,
            key=p_embds,
            value=p_embds,
            attn_mask=eye_text_attn_mask,
            need_weights=False,
        )[0]  # (batch_size, max_eye_len, text_dim)

        # concat eye_text_attn with encoded_fixations
        eye_text = torch.cat(
            (eye_text_attn, encoded_fixations), dim=2
        )  # (batch_size, max_eye_len, d_conv * 2)

        x = self.agg_fixation_by_question(eye_text, q_embds)
        output = self.classifier(x)
        # output = self.score_answers(x, a_embeds)

        return output, x.squeeze()

    def split_context_embeds(
        self,
        encoded_word_seq: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Find the positions of the separator tokens
        sep_positions = (
            ((input_ids == self.sep_token_id) | (input_ids == 41034))
            # TODO BIG HACK. Corresponds to token of "Answers" (I think; Shubi)
            .nonzero(as_tuple=True)[0]
            .cpu()
            .numpy()
        )

        # Calculate the sizes of the splits
        split_sizes = np.diff(a=sep_positions, prepend=0, append=input_ids.size(dim=0))
        assert split_sizes.sum() == input_ids.size(
            dim=0
        ), f"split_sizes.sum(): {split_sizes.sum()}"
        if self.add_answers:
            assert (
                split_sizes.size == 5
            ), f"split_sizes.size: {split_sizes.size} but expected 5"
            # Split the encoded_word_seq tensor at the separator positions
            (
                p,
                _,
                q,
                a,
                _,
            ) = torch.split(
                tensor=encoded_word_seq,
                split_size_or_sections=split_sizes.tolist(),
                dim=0,
            )
        else:
            assert (
                split_sizes.size == 4
            ), f"split_sizes.size: {split_sizes.size} but expected 4"
            # Split the encoded_word_seq tensor at the separator positions
            p, _, q, _ = torch.split(
                tensor=encoded_word_seq,
                split_size_or_sections=split_sizes.tolist(),
                dim=0,
            )
            a = torch.zeros_like(q)

        q = q[1:, :].mean(dim=0)  # TODO is the 1: correct? Why? sep?
        # a1 = a1[1:, :].mean(dim=0)
        # a2 = a2[1:, :].mean(dim=0)
        # a3 = a3[1:, :].mean(dim=0)
        # a4 = a4[1:, :].mean(dim=0)
        # a_embeds = torch.stack((a1, a2, a3, a4), dim=0)
        a_embeds = a[1:, :].mean(dim=0)
        return p, q, a_embeds

    def split_context_embds_batched(self, encoded_word_seq, input_ids):
        p_embds_batches, q_embds_batches, a_embeds_batches = [], [], []
        # Process each batch separately
        for ewsb, iib in zip(encoded_word_seq, input_ids):
            p_embds, q_embds, a_embeds = self.split_context_embeds(
                encoded_word_seq=ewsb, input_ids=iib
            )
            # pad p_embds to length of self.model_args.max_seq_len with zeros
            p_embds = F.pad(
                p_embds, (0, 0, 0, self.model_args.max_seq_len - p_embds.size(0))
            )
            p_embds_batches.append(p_embds)
            q_embds_batches.append(q_embds)
            a_embeds_batches.append(a_embeds)
        # Concatenate the results back together
        p_embds = torch.stack(p_embds_batches, dim=0)
        q_embds = torch.stack(q_embds_batches, dim=0)
        a_embeds = torch.stack(a_embeds_batches, dim=0)

        return p_embds, q_embds, a_embeds

    def agg_fixation_by_question(self, eye_text, q_embds):
        # run eye_text_attn through a linear layer so it matches the shape of q_embds
        eye_text_attn = self.project_to_text_dim(eye_text)
        return self.cross_att_agg_eyes_q(
            query=q_embds.unsqueeze(1),
            key=eye_text_attn,
            value=eye_text_attn,
            need_weights=False,
        )[0]

    def score_answers(
        self, x, a_embeds
    ) -> torch.Tensor:  # a_embeds: (2 X 4 X 1024) x: (2 X 1024)
        return torch.bmm(x, a_embeds.transpose(1, 2)).squeeze(1)
