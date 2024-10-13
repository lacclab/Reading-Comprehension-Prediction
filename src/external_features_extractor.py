from transformers import (
    RobertaModel,
)
import torch
import torch.nn.functional as F

from enum import StrEnum


class ReturnedTokensModes(StrEnum):
    CLS = "cls"
    ALL_TOKENS = "all_tokens"
    ALL_EXCEPT_CLS = "all_except_cls"


@torch.no_grad()
def encoding(
    bert_encoder: RobertaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    returned_tokens: ReturnedTokensModes,
) -> torch.Tensor:
    """
    Extract [CLS] token embeddings from the encoder.

    Args:
    - bert_encoder: RobertaModel: transformer model
    - input_ids: torch.Tensor: input ids
        shape: (batch_size, max_seq_len)
    - attention_mask: torch.Tensor: attention mask
        shape: (batch_size, max_seq_len)
    """
    # assert shape of input_ids and attention_mask
    assert input_ids.shape == attention_mask.shape
    assert len(input_ids.shape) == 2

    # assert bert_encoder on eval mode
    bert_encoder.eval()

    outputs = bert_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )  # [CLS] embeddings, shape: (batch_size, bert_dim)
    match returned_tokens:
        case ReturnedTokensModes.CLS:
            returned_embeddings = outputs.last_hidden_state[:, 0, :]
        case ReturnedTokensModes.ALL_TOKENS:
            returned_embeddings = outputs.last_hidden_state
        case ReturnedTokensModes.ALL_EXCEPT_CLS:
            returned_embeddings = outputs.last_hidden_state[:, 1:, :]
        case _:
            raise ValueError(f"Invalid value for returned_tokens: {returned_tokens}")
    return returned_embeddings


@torch.no_grad()
def compute_questions_cls_encodings(
    bert_encoder: RobertaModel,
    question_ids: torch.Tensor,
    question_masks: torch.Tensor,
) -> torch.Tensor:
    # flatten the question_ids and question_masks
    batch_size, no_questions, max_q_len = question_ids.shape
    question_ids = question_ids.view(
        -1, max_q_len
    )  # shape: (batch_size * no.questions, max_q_len)
    question_masks = question_masks.view(
        -1, max_q_len
    )  # shape: (batch_size * no.questions, max_q_len)

    # get the question_cls_encoding
    question_cls_encoding = encoding(
        bert_encoder=bert_encoder,
        input_ids=question_ids,
        attention_mask=question_masks,
        returned_tokens=ReturnedTokensModes.CLS,
    )  # shape: (batch_size * no.questions, bert_dim)
    bert_dim = question_cls_encoding.shape[-1]

    # reshape the question_cls_encoding
    question_cls_encoding = question_cls_encoding.view(
        batch_size, no_questions, bert_dim
    )

    return question_cls_encoding  # shape: (batch_size, no.questions, bert_dim)


@torch.no_grad()
def move_to_device(device, *tensors):
    return (tensor.to(device) for tensor in tensors)


@torch.no_grad()
def feature_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT(
    device: str,
    bert_encoder: RobertaModel,
    question_ids: torch.Tensor,
    question_masks: torch.Tensor,
    paragraph_ids: torch.Tensor,
    paragraph_masks: torch.Tensor,
    dwell_time_weights: torch.Tensor,
) -> torch.Tensor:
    """
    dot product between, 1) question [CLS] token embedding
    and 2) np.average of word embeddings of the paragraph, weighted by IA_DWELL_TIME.

    no context refers to the fact that the paragraph and the questions
    are passed separately through the encoder.

    Args:
    - device: str: device to run the model on
    - bert_encoder: RobertaModel: transformer model  #! should be on the device
    - question_ids: torch.Tensor: question input ids
        shape: (batch_size, no.questions, max_q_len)
    - question_masks: torch.Tensor: question attention masks
        shape: (batch_size, no.questions, max_q_len)
    - paragraph_ids: torch.Tensor: paragraph input ids
        shape: (batch_size, max_seq_len)
    - paragraph_masks: torch.Tensor: paragraph attention masks
        shape: (batch_size, max_seq_len)
    - dwell_time_weights: torch.Tensor: IA_DWELL_TIME weights
        shape: (batch_size, no.questions, ~max_seq_len)

    Returns:
    - questions_scores: torch.Tensor: dot product between question [CLS] embeddings
        and weighted average paragraph embeddings
        shape: (batch_size, no.questions)
    """
    # assert shape of question_ids and question_masks
    assert question_ids.shape == question_masks.shape
    assert len(question_ids.shape) == 3

    # assert shape of paragraph_ids and paragraph_masks
    assert paragraph_ids.shape == paragraph_masks.shape
    assert len(paragraph_ids.shape) == 2

    # assert shape of dwell_time_weights
    assert dwell_time_weights.shape[0] == question_ids.shape[0]
    assert dwell_time_weights.shape[1] == question_ids.shape[1]

    # move all tensors to device
    (
        question_ids,
        question_masks,
        paragraph_ids,
        paragraph_masks,
        dwell_time_weights,
    ) = move_to_device(
        device,
        question_ids,
        question_masks,
        paragraph_ids,
        paragraph_masks,
        dwell_time_weights,
    )

    questions_cls_encodings = compute_questions_cls_encodings(
        bert_encoder=bert_encoder,
        question_ids=question_ids,
        question_masks=question_masks,
    )  # shape: (batch_size, no.questions, bert_dim)

    # get paragraph per word embeddings
    paragraph_tokens_embeddings = encoding(
        bert_encoder=bert_encoder,
        input_ids=paragraph_ids,
        attention_mask=paragraph_masks,
        returned_tokens=ReturnedTokensModes.ALL_TOKENS,
    )  # shape: (batch_size, max_seq_len, bert_dim)

    # pad dwell_time_weights based on paragraph_masks
    dwell_empty_value = float(
        dwell_time_weights[0, 0, 0].clone().detach()
    )  # should be 0.0
    dwell_pad_length = paragraph_masks.shape[-1] - dwell_time_weights.shape[-1]

    # pad dwell_time_weights
    dwell_time_weights = torch.nn.functional.pad(
        dwell_time_weights, (0, dwell_pad_length), value=dwell_empty_value
    )  # shape: (batch_size, no.questions, max_seq_len)

    # Compute dwell time weighted average of paragraph embeddings
    dwell_time_weights = dwell_time_weights.unsqueeze(
        -1
    )  # add an extra dimension for broadcasting, shape: (batch_size, no.questions, max_seq_len, 1)
    paragraph_tokens_embeddings = paragraph_tokens_embeddings.unsqueeze(
        1
    )  # shape: (batch_size, 1, max_seq_len, bert_dim)
    weighted_paragraph_embeddings = (
        paragraph_tokens_embeddings * dwell_time_weights
    )  # shape: (batch_size, no.questions, max_seq_len, bert_dim)
    weighted_average_paragraph_embeddings = weighted_paragraph_embeddings.sum(
        dim=2
    ) / dwell_time_weights.sum(
        dim=2
    )  # shape: (batch_size, no.questions, bert_dim)

    # Compute dot product between question [CLS] embeddings and weighted average paragraph embeddings
    # shapes: (batch_size, no.questions, bert_dim) and (batch_size, no.questions, bert_dim)
    questions_scores = (
        questions_cls_encodings * weighted_average_paragraph_embeddings
    ).sum(
        dim=-1
    )  # shape: (batch_size, no.questions)

    return questions_scores


@torch.no_grad()
def feature_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT(
    device: str,
    bert_encoder: RobertaModel,
    question_ids: torch.Tensor,
    question_masks: torch.Tensor,
    paragraph_ids: torch.Tensor,
    paragraph_masks: torch.Tensor,
    dwell_time_weights: torch.Tensor,
) -> torch.Tensor:
    """
    1) cosine similarity between question [CLS] token embedding and paragraph's token embeddings
    2) dot product of (1) with IA_DWELL_TIME

    no context refers to the fact that the paragraph and the questions
    are passed separately through the encoder.

    Args:
    - device: str: device to run the model on
    - bert_encoder: RobertaModel: transformer model  #! should be on the device
    - question_ids: torch.Tensor: question input ids
        shape: (batch_size, no.questions, max_q_len)
    - question_masks: torch.Tensor: question attention masks
        shape: (batch_size, no.questions, max_q_len)
    - paragraph_ids: torch.Tensor: paragraph input ids
        shape: (batch_size, max_seq_len)
    - paragraph_masks: torch.Tensor: paragraph attention masks
        shape: (batch_size, max_seq_len)
    - dwell_time_weights: torch.Tensor: IA_DWELL_TIME weights
        shape: (batch_size, no.questions, ~max_seq_len)

    Returns:
    - questions_scores: torch.Tensor: dot product between question [CLS] embeddings
        and weighted average paragraph embeddings
        shape: (batch_size, no.questions)
    """
    # assert shape of question_ids and question_masks
    assert question_ids.shape == question_masks.shape
    assert len(question_ids.shape) == 3

    # assert shape of paragraph_ids and paragraph_masks
    assert paragraph_ids.shape == paragraph_masks.shape
    assert len(paragraph_ids.shape) == 2

    # assert shape of dwell_time_weights
    assert dwell_time_weights.shape[0] == question_ids.shape[0]
    assert dwell_time_weights.shape[1] == question_ids.shape[1]

    # move all tensors to device
    (
        question_ids,
        question_masks,
        paragraph_ids,
        paragraph_masks,
        dwell_time_weights,
    ) = move_to_device(
        device,
        question_ids,
        question_masks,
        paragraph_ids,
        paragraph_masks,
        dwell_time_weights,
    )

    questions_cls_encodings = compute_questions_cls_encodings(
        bert_encoder=bert_encoder,
        question_ids=question_ids,
        question_masks=question_masks,
    )  # shape: (batch_size, no.questions, bert_dim)

    # get paragraph per word embeddings
    paragraph_tokens_embeddings = encoding(
        bert_encoder=bert_encoder,
        input_ids=paragraph_ids,
        attention_mask=paragraph_masks,
        returned_tokens=ReturnedTokensModes.ALL_TOKENS,
    )  # shape: (batch_size, max_seq_len, bert_dim)

    # multiply paragraph_tokens_embeddings with question_cls_encodings
    # out shape: (batch_size, no.questions, max_seq_len)
    paragraph_tokens_embeddings = paragraph_tokens_embeddings.unsqueeze(
        1
    )  # shape: (batch_size, 1, max_seq_len, bert_dim)
    questions_cls_encodings = questions_cls_encodings.unsqueeze(
        -2
    )  # shape: (batch_size, no.questions, 1, bert_dim)
    # cosine similarity
    questions_rel_span = F.cosine_similarity(
        paragraph_tokens_embeddings, questions_cls_encodings, dim=-1
    )  # shape: (batch_size, no.questions, max_seq_len)

    # pad dwell_time_weights based on paragraph_masks
    dwell_empty_value = float(
        dwell_time_weights[0, 0, 0].clone().detach()
    )  # should be 0.0
    dwell_pad_length = paragraph_masks.shape[-1] - dwell_time_weights.shape[-1]

    # pad dwell_time_weights
    dwell_time_weights = torch.nn.functional.pad(
        dwell_time_weights, (0, dwell_pad_length), value=dwell_empty_value
    )  # shape: (batch_size, no.questions, max_seq_len)

    # Compute dot product between questions_rel_span and dwell_time_weights
    # shapes: (batch_size, no.questions, max_seq_len) and (batch_size, no.questions, max_seq_len)
    questions_scores = (questions_rel_span * dwell_time_weights).sum(
        dim=-1
    )  # shape: (batch_size, no.questions)

    return questions_scores
