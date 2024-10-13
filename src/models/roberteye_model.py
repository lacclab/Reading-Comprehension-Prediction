"""roberteye.py"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    RobertaForMultipleChoice,
    RobertaForSequenceClassification,
    RobertaModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaPooler,
    create_position_ids_from_input_ids,
)

from src.configs.constants import DataRepresentation, PredMode
from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.model_specific_args.RoBERTeyeArgs import Roberteye
from src.configs.trainer_args import Base
from src.models.base_roberta import BaseMultiModalRoberta


@dataclass
class MultimodalConfig:
    text_dim: int
    eyes_dim: int
    dropout: float


class RoBERTeyeModel(BaseMultiModalRoberta):
    """
    Model for Multiple Choice Question Answering and question prediction tasks.
    """

    def __init__(
        self,
        model_args: Roberteye,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ):
        super().__init__(model_args, trainer_args)
        print(f"Is training: {model_args.is_training}")
        self.multimodal_config = MultimodalConfig(
            text_dim=model_args.text_dim,
            eyes_dim=(
                model_args.fixation_dim
                if model_args.use_fixation_report
                else model_args.eyes_dim
            ),
            dropout=model_args.eye_projection_dropout,
        )

        if (
            self.prediction_mode
            in [
                PredMode.QUESTION_LABEL,
                PredMode.CHOSEN_ANSWER,
                PredMode.CORRECT_ANSWER,
                PredMode.CONDITION,
                PredMode.IS_CORRECT,
                PredMode.QUESTION_n_CONDITION,
            ]
            and model_args.model_params.concat_or_duplicate == DataRepresentation.CONCAT
        ):
            if model_args.is_training:
                model = RobertEyeForSequenceClassification.from_pretrained(
                    model_args.backbone,
                    num_labels=self.num_classes,
                    multimodal_config=self.multimodal_config,
                )
                model = adjust_model_for_eyes(
                    model,  # type: ignore
                    eye_token_id=model_args.eye_token_id,
                    sep_token_id=model_args.sep_token_id,
                )
                self.model = model
            else:
                robertaconfig = RobertaConfig.from_pretrained(
                    model_args.backbone,
                    vocab_size=50266,
                    type_vocab_size=2,
                    num_labels=self.num_classes,
                )

                self.model = RobertEyeForSequenceClassification(
                    config=robertaconfig, multimodal_config=self.multimodal_config
                )

        elif (
            self.prediction_mode
            in [
                PredMode.QUESTION_LABEL,
                PredMode.CHOSEN_ANSWER,
                PredMode.CORRECT_ANSWER,
                PredMode.QUESTION_n_CONDITION,
            ]
            and model_args.model_params.concat_or_duplicate
            == DataRepresentation.DUPLICATE
        ):  # Based on https://github.com/Lightning-AI/pytorch-lightning/discussions/9236
            if model_args.is_training:
                model = RobertEyeForMultipleChoice.from_pretrained(
                    model_args.backbone,
                    multimodal_config=self.multimodal_config,
                )
                model = adjust_model_for_eyes(
                    model,  # type: ignore
                    eye_token_id=model_args.eye_token_id,
                    sep_token_id=model_args.sep_token_id,
                )
                self.model = model

            else:
                robertaconfig = RobertaConfig.from_pretrained(
                    model_args.backbone, vocab_size=50266, type_vocab_size=2
                )

                self.model = RobertEyeForMultipleChoice(
                    config=robertaconfig, multimodal_config=self.multimodal_config
                )

        else:
            raise ValueError(
                f"Invalid combination: prediction_mode - {self.prediction_mode}, "
                f"concat_or_duplicate - {model_args.model_params.concat_or_duplicate}"
            )

        if model_args.freeze:
            # Freeze all model parameters except specific ones
            for name, param in self.named_parameters():
                if (
                    name.startswith("model.roberta.embeddings.eye_projection")
                    or name.startswith(
                        "model.roberta.embeddings.eye_position_embeddings"
                    )
                    or name.startswith("model.roberta.embeddings.token_type_embeddings")
                    or name.startswith("model.classifier")
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.save_hyperparameters()


class RoberteyeEmbeddings(RobertaEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    Based on https://github.com/uclanlp/visualbert/tree/master/visualbert
    """

    def __init__(self, config: RobertaConfig, multimodal_config: MultimodalConfig):
        super().__init__(config)
        # Token type and position embedding for eye features
        self.eye_position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.eye_position_embeddings.weight.data = nn.Parameter(
            self.position_embeddings.weight.data.clone(), requires_grad=True
        )
        projection_dropout = multimodal_config.dropout
        self.eye_projection = nn.Sequential(
            nn.Linear(
                in_features=multimodal_config.eyes_dim,
                out_features=config.hidden_size // 2,
            ),
            nn.ReLU(),
            nn.Dropout(p=projection_dropout),
            nn.Linear(
                in_features=config.hidden_size // 2, out_features=config.hidden_size
            ),
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        eye_embeds=None,
        eye_token_type_ids=None,
        eye_position_ids=None,
        eye_positions=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]  # type: ignore
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(  # type: ignore
                    input_shape,
                    dtype=torch.long,
                    device=self.position_ids.device,  # type: ignore
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Eye movements addition
        if eye_embeds is not None:
            if eye_token_type_ids is None:
                eye_token_type_ids = torch.ones(  # type: ignore
                    eye_embeds.size()[:-1],
                    dtype=torch.long,
                    device=self.position_ids.device,  # type: ignore
                )

            eye_embeds = self.eye_projection(eye_embeds)
            eye_token_type_embeddings = self.token_type_embeddings(eye_token_type_ids)

            # image_text_alignment = Batch x image_length x alignment_number.
            # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

            dtype = token_type_embeddings.dtype
            eyes_text_alignment_mask = (eye_positions != -1).long()
            # Get rid of the -1.
            eye_positions = eyes_text_alignment_mask * eye_positions

            # Batch x image_length x alignment length x dim
            eye_position_embeddings = self.position_embeddings(eye_positions)
            eye_position_embeddings *= eyes_text_alignment_mask.to(
                dtype=dtype
            ).unsqueeze(-1)
            eye_position_embeddings = eye_position_embeddings.sum(2)

            # We want to average along the alignment_number dimension.
            eyes_text_alignment_mask = eyes_text_alignment_mask.to(dtype=dtype).sum(2)

            if (eyes_text_alignment_mask == 0).sum() != 0:
                eyes_text_alignment_mask[
                    eyes_text_alignment_mask == 0
                ] = 1  # Avoid divide by zero error
                # print(
                #     "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero"
                #     " error."
                # )
            eye_position_embeddings = (
                eye_position_embeddings / eyes_text_alignment_mask.unsqueeze(-1)
            )

            # visual_position_ids = torch.zeros(
            #     *eye_embeds.size()[:-1], dtype=torch.long, device=eye_embeds.device
            # )

            # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
            if eye_position_embeddings.size(1) != eye_embeds.size(1):
                if eye_position_embeddings.size(1) < eye_embeds.size(1):
                    raise ValueError(
                        f"Visual position embeddings length: {eye_position_embeddings.size(1)} "
                        f"should be the same as `eye_embeds` length: {eye_embeds.size(1)}"
                    )
                eye_position_embeddings = eye_position_embeddings[
                    :, : eye_embeds.size(1), :
                ]

            # eye_position_embeddings = eye_position_embeddings + self.eye_position_embeddings(
            #     visual_position_ids
            # )

            # if eye_position_ids is None:
            #     eye_position_ids = create_position_ids_from_input_ids(
            #         eye_positions,
            #         self.padding_idx,
            #         past_key_values_length,
            #     )
            # eye_position_embeddings = self.eye_position_embeddings(eye_position_ids)

            final_eye_embeds = (
                eye_embeds + eye_position_embeddings + eye_token_type_embeddings
            )

            cls_token, rest_embedding_output = (
                embeddings[:, 0:1, :],
                embeddings[:, 1:, :],
            )
            # Concatenate the CLS token, eye, and the rest of the embedding_output
            embeddings = torch.cat(
                (cls_token, final_eye_embeds, rest_embedding_output), dim=1
            )
            # Final format: CLS EYES EYE_TOKEN SEP_TOKEN REST_OF_THE_TEXT
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RoBERTeyeEncoderModel(RobertaModel):
    """
    This class is a modified version of the RobertaModel class from the transformers library.
    It adds the MAG module to the forward pass.
    """

    def __init__(
        self, config, multimodal_config: MultimodalConfig, add_pooling_layer=True
    ):
        super().__init__(config, add_pooling_layer)
        self.config = config

        self.embeddings = RoberteyeEmbeddings(config, multimodal_config)
        self.encoder = RobertaEncoder(config)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.forward
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        gaze_features: torch.Tensor | None = None,
        gaze_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape
        `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention if the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape
        `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input.
            This mask is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with
        each tuple having 4 tensors of shape
        `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks.
            Can be used to speed up decoding.

            If `past_key_values` are used, can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape
            `(batch_size, 1)` instead of all `decoder_input_ids` of
            shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned
            and can be used to speed up decoding (see `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if gaze_features is not None:
            input_shape = torch.Size(
                (input_shape[0], input_shape[1] + gaze_features.size()[1])
            )

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=self.device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                token_seq_length = input_ids.size()[1]  # type: ignore
                buffered_token_type_ids = self.embeddings.token_type_ids[  # type: ignore
                    :, :token_seq_length
                ]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, token_seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.device
                )

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask,
            input_shape,  # type: ignore
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=self.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            eye_embeds=gaze_features,
            eye_positions=gaze_positions,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,  # type: ignore
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertEyeForSequenceClassification(RobertaForSequenceClassification):
    """
    This class is a modified version of the RobertaForSequenceClassification class
    from the transformers library.

    Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RoBERTeyeEncoderModel(
            config, multimodal_config, add_pooling_layer=False
        )
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        gaze_features: torch.Tensor | None = None,
        gaze_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: torch.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.roberta(
            input_ids=input_ids,
            gaze_features=gaze_features,
            gaze_positions=gaze_positions,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertEyeForMultipleChoice(RobertaForMultipleChoice):
    """
    This class is a modified version of the RobertaForMultipleChoice class
    from the transformers library.
    It adds the MAG module to the forward pass.

    Copied from transformers.models.roberta.modeling_roberta.RobertaForMultipleChoice
    """

    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.roberta = RoBERTeyeEncoderModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        gaze_features: torch.Tensor | None = None,
        gaze_positions: torch.Tensor | None = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> MultipleChoiceModelOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss.
            Indices should be in `[0, ..., num_choices-1]` where `num_choices`
            is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        num_choices = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]  # type: ignore
        )

        flat_input_ids = (
            input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        )
        flat_position_ids = (
            position_ids.view(-1, position_ids.size(-1))
            if position_ids is not None
            else None
        )
        flat_token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1))
            if token_type_ids is not None
            else None
        )
        flat_attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1))
            if attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        flat_gaze_features = (
            gaze_features.view(-1, gaze_features.size(-2), gaze_features.size(-1))
            if gaze_features is not None
            else None
        )
        flat_gaze_positions = (
            gaze_positions.view(-1, gaze_positions.size(-2), gaze_positions.size(-1))
            if gaze_positions is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            flat_gaze_features,
            gaze_positions=flat_gaze_positions,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(reshaped_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def adjust_model_for_eyes(
    model: RobertEyeForMultipleChoice | RobertEyeForSequenceClassification,
    eye_token_id: int,
    sep_token_id: int,
) -> RobertEyeForMultipleChoice | RobertEyeForSequenceClassification:
    model.config.vocab_size += 1  # type: ignore
    # Add token_type+1 (for eye token id (=1) and token_embedding+1 (for Eye SEP)
    # Add 1 to the vocab size for the eye token
    # https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
    model.resize_token_embeddings(model.config.vocab_size)  # type: ignore

    # Itialize the eye token embedding to the SEP token embedding
    with torch.no_grad():
        model.roberta.embeddings.word_embeddings.weight[eye_token_id] = (  # type: ignore
            model.roberta.embeddings.word_embeddings.weight[sep_token_id]  # type: ignore
            .detach()
            .clone()
        )

    model.config.type_vocab_size += 1  # type: ignore

    single_emb: nn.Embedding = (
        model.roberta.embeddings.token_type_embeddings  # type: ignore
    )

    model.roberta.embeddings.token_type_embeddings = nn.Embedding(  # type: ignore
        model.config.type_vocab_size,  # type: ignore
        single_emb.embedding_dim,
    )

    # https://github.com/huggingface/transformers/issues/1538#issuecomment-570260748
    model.roberta.embeddings.token_type_embeddings.weight = (  # type: ignore
        torch.nn.Parameter(single_emb.weight.repeat([2, 1]))
    )
    return model
