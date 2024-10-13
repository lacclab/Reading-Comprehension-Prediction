"""Eyettention model
Based on https://github.com/aeye-lab/Eyettention/tree/main
"""

import string
from typing import List

from pytorch_metric_learning import losses
import torch
from matplotlib import pyplot as plt
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,
)

from src.configs.constants import MAX_SCANPATH_LENGTH, PredMode
from src.configs.data_args import DataArgs
from src.configs.data_path_args import DataPathArgs
from src.configs.model_args.model_specific_args.EyettentionArgs import EyettentionArgs
from src.configs.trainer_args import Base
from src.models.base_model import BaseModel


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def pad_list(input_list, target_length, pad_with=0):
    # Calculate how many elements need to be added
    padding_length = target_length - len(input_list)

    # If padding_length is less than 0, the list is already longer than target_length
    if padding_length < 0:
        print("The list is already longer than the target length.")
        return input_list

    # Add padding_length number of zeros to the end of the list
    padded_list = input_list + [pad_with] * padding_length

    return padded_list


def get_word_length(word):
    if word in ["<s>", "</s>", "<pad>"]:
        return 0
    else:
        return len(word.translate(str.maketrans("", "", string.punctuation)))


def align_word_ids_with_input_ids(
    tokenizer: RobertaTokenizerFast,
    input_ids: torch.Tensor,
    decoded_to_txt_input_ids: list,
):
    word_ids_sn_lst = []
    retokenized_sn = tokenizer(
        decoded_to_txt_input_ids,
        return_tensors="pt",
    )
    for i in range(input_ids.shape[0]):
        word_ids_sn_lst.append(retokenized_sn.word_ids(i)[1:-1])

    word_ids_sn = torch.tensor(word_ids_sn_lst).to(input_ids.device)

    return word_ids_sn


def get_sn_word_lens(input_ids: torch.Tensor, decoded_to_txt_input_ids: list):
    def compute_p_lengths(p, target_length):
        return pad_list([get_word_length(word) for word in p], target_length)

    target_len = input_ids.shape[1]
    sn_word_len = torch.tensor(
        [
            compute_p_lengths(paragraph, target_len)
            for paragraph in decoded_to_txt_input_ids
        ]
    ).to(input_ids.device)

    return sn_word_len


def convert_positions_to_words_sp(
    scanpath: torch.Tensor,
    decoded_to_txt_input_ids: List[List[str]],
    roberta_tokenizer_prefix_space: RobertaTokenizerFast,
):
    sp_tokens_strs = []
    for i in range(scanpath.shape[0]):
        curr_sp_tokens = [roberta_tokenizer_prefix_space.cls_token] + [
            decoded_to_txt_input_ids[i][word_i + 1]  # +1 to skip the <s> token
            for word_i in scanpath[i].tolist()
            if word_i != -1
        ]
        curr_sp_tokens_str = " ".join(curr_sp_tokens)
        sp_tokens_strs.append(curr_sp_tokens_str.split())

    return sp_tokens_strs


def calc_sp_word_input_ids(
    input_ids: torch.Tensor,
    decoded_to_txt_input_ids: List[List[str]],
    backbone: str,
    scanpath: torch.Tensor,
):
    """This function calculates the word input ids for the scanpath

    Args:
        input_ids (torch.Tensor): The word sequence input ids.
                Tensor of (batch_size, max_text_length_in_tokens)
        decoded_to_txt_input_ids (list): The decoded input ids.
                (list of lists of strings)
        max_eye_len (int): The maximum scanpath length in the current batch (not global)
        backbone (str):  The backbone of the model (roberta base/large/RACE)
        scanpath (torch.Tensor): A scanpath tensor containing the word indices in the scanpath order
                Tensor of (batch_size, max_scanpath_length_in_words)
    """
    SP_word_ids, SP_input_ids = [], []
    roberta_tokenizer_prefix_space = RobertaTokenizerFast.from_pretrained(
        backbone, add_prefix_space=True
    )

    sp_tokens_strs = convert_positions_to_words_sp(
        scanpath=scanpath,
        decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        roberta_tokenizer_prefix_space=roberta_tokenizer_prefix_space,
    )

    tokenized_SPs = roberta_tokenizer_prefix_space.batch_encode_plus(
        sp_tokens_strs,
        add_special_tokens=False,
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        is_split_into_words=True,
    )
    for i in range(scanpath.shape[0]):
        encoded_sp = tokenized_SPs["input_ids"][i]
        word_ids_sp = tokenized_SPs.word_ids(i)  # -> Take the <sep> into account
        word_ids_sp = [val if val is not None else -1 for val in word_ids_sp]

        SP_word_ids.append(word_ids_sp)
        SP_input_ids.append(encoded_sp)

    word_ids_sp = torch.tensor(SP_word_ids).to(input_ids.device)
    sp_input_ids = torch.tensor(SP_input_ids).to(input_ids.device)

    return word_ids_sp, sp_input_ids


class EyettentionModel(BaseModel):
    """Eyettention model"""

    # TODO move class to top of file

    def __init__(
        self,
        model_args: EyettentionArgs,
        trainer_args: Base,
        data_args: DataArgs,
        data_path_args: DataPathArgs,
    ) -> None:
        super().__init__(model_args, trainer_args)
        # self.emodel = self.Eyettention_original()
        self.model_args = model_args
        self.backbone = model_args.backbone
        self.preorder = False

        self.bert_dim = model_args.text_dim
        self.hidden_size = model_args.model_params.LSTM_hidden_dim
        self.max_seq_len = model_args.max_seq_len
        self.max_eye_len = model_args.max_eye_len

        # Word-Sequence Encoder
        encoder_config = RobertaConfig.from_pretrained(self.backbone)
        encoder_config.output_hidden_states = True
        # initiate Bert with pre-trained weights
        print("keeping Bert with pre-trained weights")
        self.bert_encoder: RobertaModel = RobertaModel.from_pretrained(
            self.backbone, config=encoder_config
        )  # type: ignore

        # freeze the parameters in Bert model
        # TODO Replace for with with torch nograd and eval()?
        for param in self.bert_encoder.parameters():  # type: ignore
            param.requires_grad = False

        self.embedding_dropout = nn.Dropout(model_args.model_params.embedding_dropout)
        self.encoder_lstm = nn.LSTM(
            input_size=self.bert_dim,
            hidden_size=int(self.hidden_size / 2),
            num_layers=model_args.model_params.num_LSTM_layers,
            batch_first=True,
            bidirectional=True,
            dropout=model_args.model_params.LSTM_dropout,
        )

        self.position_embeddings = nn.Embedding(
            encoder_config.max_position_embeddings, encoder_config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(
            encoder_config.hidden_size, eps=encoder_config.layer_norm_eps
        )

        # Cross-Attention
        # self.attn = nn.Linear(
        #     self.hidden_size, self.hidden_size + 1
        # )  # +1 acount for the word length feature

        self.cross_attention_layer = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=1,
            dropout=0.2,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=self.hidden_size + 1,
            vdim=self.hidden_size + 1,
            batch_first=True,
        )

        decoder_dropout = model_args.model_params.fc_dropout
        self.decoder_dense = nn.Sequential(
            nn.Dropout(decoder_dropout),
            nn.Linear(
                self.hidden_size * 2, 512
            ),  #! was self.hidden_size * 2 before I changed the cross attention
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

        self.wse_to_fc = nn.Sequential(
            nn.Dropout(decoder_dropout),
            nn.Linear(self.hidden_size + 1, self.hidden_size * 2),
            nn.ReLU(),
        )

        self.fse_lstm = nn.LSTM(
            input_size=self.bert_dim
            + 2,  # word embd + Total fixtion duration + Landing position
            hidden_size=self.hidden_size,
            num_layers=model_args.model_params.num_LSTM_layers,
            dropout=model_args.model_params.LSTM_dropout,
            batch_first=True,
        )

        self.fast_tokenizer = RobertaTokenizerFast.from_pretrained(self.backbone)
        self.pad_token_id = self.fast_tokenizer.pad_token_id

        if self.model_args.add_contrastive_loss:
            self.cl_alpha = 1
            self.cl_memory_size = self.hidden_size * 2
            self.cl_temperature = 0.07
            self.contrastive_loss = losses.CrossBatchMemory(
                loss=losses.NTXentLoss(temperature=self.cl_temperature),
                embedding_size=self.hidden_size * 2,
                memory_size=self.cl_memory_size,
            )
            print(f"##### Contrastive loss added with alpha: {self.cl_alpha} #####")
            print(f"##### Contrastive loss memory size: {self.cl_memory_size} #####")
            print(f"##### Contrastive loss temperature: {self.cl_temperature} #####")
            print(
                f"##### Contrastive loss embedding size: {self.model_args.text_dim} #####"
            )

        self.save_hyperparameters()

    def pool_subword_to_word(self, subword_emb, word_ids_sn, target, pool_method="sum"):
        # batching computing
        # Pool bert token (subword) to word level
        if target == "sn":
            max_len = self.max_seq_len  # CLS and SEP included
        elif target == "sp":
            max_len = (
                word_ids_sn.max().item() + 1
            )  # +1 for the <s> token at the beginning
        else:
            raise NotImplementedError

        merged_word_emb = torch.empty(subword_emb.shape[0], 0, self.bert_dim).to(
            subword_emb.device
        )
        for word_idx in range(max_len):
            word_mask = (
                (word_ids_sn == word_idx)
                .unsqueeze(2)
                .repeat(1, 1, self.bert_dim)
                .to(subword_emb.device)
            )
            # pooling method -> sum
            if pool_method == "sum":
                pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(
                    1
                )  # [batch, 1, 1024]
            elif pool_method == "mean":
                pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(
                    1
                )  # [batch, 1, 1024]
            else:
                raise NotImplementedError
            merged_word_emb = torch.cat([merged_word_emb, pooled_word_emb], dim=1)

        mask_word = torch.sum(merged_word_emb, 2).bool()
        return merged_word_emb, mask_word

    def word_sequence_encoder(self, sn_emd, sn_mask, word_ids_sn, sn_word_len):
        with torch.no_grad():
            outputs = self.bert_encoder(input_ids=sn_emd, attention_mask=sn_mask)
        #  Make the embedding of the <pad> token to be zeros
        outputs.last_hidden_state[sn_emd == self.pad_token_id] = 0

        # Pool bert subword to word level for english corpus
        merged_word_emb, sn_mask_word = self.pool_subword_to_word(
            outputs.last_hidden_state, word_ids_sn, target="sn", pool_method="sum"
        )

        merged_word_emb = self.embedding_dropout(merged_word_emb)
        packed_merged_word_emb = nn.utils.rnn.pack_padded_sequence(
            input=merged_word_emb,
            lengths=torch.sum(sn_mask_word, dim=1).to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        assert not torch.isnan(packed_merged_word_emb.data).any()

        x, _ = self.encoder_lstm(packed_merged_word_emb)
        x = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, total_length=merged_word_emb.shape[1]
        )[0]

        # concatenate with the word length feature
        x = torch.cat((x, sn_word_len.unsqueeze(2)), dim=2)

        return x, sn_mask_word

    def cross_attention(self, ht, hs, sn_mask):
        # TODO replace with attention layer?
        # TODO https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        # General Attention:
        # score(ht,hs) = (ht^T)(Wa)hs
        # hs is the output from word-Sequence Encoder
        # ht is the previous hidden state from Fixation-Sequence Encoder
        # self.attn(o): [batch, step, units]

        attention_res = self.cross_attention_layer(
            query=ht.unsqueeze(1),
            key=hs,
            value=hs,
            key_padding_mask=~sn_mask.bool(),
        )
        return attention_res[0].squeeze(1)

    def fixation_sequence_encoder(
        self, sp_emd, sp_pos, sp_fix_dur, sp_landing_pos, word_ids_sp
    ):
        """A LSTM based encoder for the fixation sequence (scanpath)
        Args:
            sp_emd (torch.Tensor): A tensor containing the text input_ids ordered according to the scanpath
            sp_pos (torch.Tensor): The word index of each fixation in the scanpath (the word the fixation is on)
            sp_fix_dur (torch.Tensor): The total fixation duration of each word in the scanpath (fixation)
            sp_landing_pos (torch.Tensor): The landing position of each word in the scanpath (fixation)
            word_ids_sp (torch.Tensor): The word index of each input_id in the scanpath

        Returns:
            _type_: _description_
        """
        x = self.bert_encoder.embeddings.word_embeddings(sp_emd)
        x[sp_emd == self.pad_token_id] = 0
        # Pool bert subword to word level for English corpus
        sp_merged_word_emd, sp_merged_word_mask = self.pool_subword_to_word(
            x, word_ids_sp, target="sp", pool_method="sum"
        )

        # add positional embeddings
        position_embeddings = self.position_embeddings(sp_pos)
        x = sp_merged_word_emd + position_embeddings
        x = self.layer_norm(x)
        x = x.permute(1, 0, 2)  # [step, n, emb_dim]
        x = self.embedding_dropout(x)

        # concatenate two additional gaze features
        x = torch.cat((x, sp_fix_dur.permute(1, 0)[:, :, None]), dim=2)
        x = torch.cat((x, sp_landing_pos.permute(1, 0)[:, :, None]), dim=2)
        x = x.permute(1, 0, 2)  # [batch, step, emb_dim]

        # pass through the LSTM layer
        sorted_lengths, indices = torch.sort(
            torch.sum(sp_merged_word_mask, dim=1), descending=True
        )
        x = x[indices]

        # Pass the entire sequence through the LSTM layer
        packed_x = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=sorted_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=True,
        )

        packed_output, (ht, ct) = self.fse_lstm(packed_x)
        lstm_last_hidden = ht[
            -1
        ].squeeze()  # Take the hidden state of the 8th LSTM layer.

        # reorder the hidden states to the original order
        lstm_last_hidden = lstm_last_hidden[
            torch.argsort(indices)
        ]  # Tested. Reorders correctly

        return lstm_last_hidden  # Take the hidden state of the 8th LSTM layer.

    def forward(
        self,
        sn_emd,
        sn_mask,
        word_ids_sn,
        sn_word_len,
        sp_emd,  # (Batch, Maximum length of the scanpath in TOKENS + 1)
        sp_pos,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        sp_fix_dur,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        sp_landing_pos,  # (Batch, Scanpath_length + 1) The +1 is for the <s> token in the beginning
        word_ids_sp,  # (Batch, Maximum length of the scanpath in TOKENS + 1)
    ):
        assert (
            sn_emd[:, 0].sum().item() == 0
        )  # The CLS token is always present first (and 0 in roberta)
        wse_output, _ = self.word_sequence_encoder(
            sn_emd, sn_mask, word_ids_sn, sn_word_len
        )  # [batch, step, units], [batch, units]

        fse_output = self.fixation_sequence_encoder(
            sp_emd,
            sp_pos,
            sp_fix_dur,
            sp_landing_pos,
            word_ids_sp,
        )  # [batch, step, dec_o_dim]

        # Apply cross attention only on the final output
        context = self.cross_attention(ht=fse_output, hs=wse_output, sn_mask=sn_mask)

        # Decoder
        hc = torch.cat([context, fse_output], dim=1)  # [batch, units *2]

        pred = self.decoder_dense(hc)

        return pred, hc

    def shared_step(
        # TODO update similar to base_roberta.py for ordered classification
        self,
        batch: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            batch (tuple): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_

        Notes:
            - in input ids: 0 is for CLS, 2 is for SEP, 1 is for PAD
        """
        # (paragraph_input_ids,
        # paragraph_input_masks,
        # input_ids,
        # input_masks,
        # labels,
        # eyes,
        # answer_mappings,
        # fixation_features,
        # fixation_pads,
        # scanpath, #? Scanpath is in IA_IDs format, need to turn it to input_ids format using inversions and input ids
        # scanpath_pads,
        # inversions,
        # inversions_pads,
        # grouped_inversions,
        # trial_level_features)
        batch_data = self.unpack_batch(batch)
        assert batch_data.input_ids is not None, "input_ids not in batch_dict"
        assert batch_data.input_masks is not None, "input_masks not in batch_dict"
        assert batch_data.scanpath is not None, "scanpath not in batch_dict"
        assert batch_data.fixation_features is not None, "eyes_tensor not in batch_dict"
        assert batch_data.scanpath_pads is not None, "scanpath_pads not in batch_dict"

        shortest_scanpath_pad = batch_data.scanpath_pads.min()
        longest_batch_scanpath: int = int(MAX_SCANPATH_LENGTH - shortest_scanpath_pad)

        scanpath = batch_data.scanpath[..., :longest_batch_scanpath]
        fixation_features = batch_data.fixation_features[
            ..., :longest_batch_scanpath, :
        ]

        decoded_to_txt_input_ids = self.fast_tokenizer.batch_decode(
            batch_data.input_ids, return_tensors="pt"
        )

        word_ids_sn = align_word_ids_with_input_ids(
            tokenizer=self.fast_tokenizer,
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        )

        # in the decoded texts, space between <pad><pad>, <pad><s>, etc.
        decoded_to_txt_input_ids = list(
            map(
                lambda x: x.replace("<", " <").split(" ")[1:],
                decoded_to_txt_input_ids,
            )
        )

        sn_word_len = get_sn_word_lens(
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
        )

        word_ids_sp, sp_input_ids = calc_sp_word_input_ids(
            input_ids=batch_data.input_ids,
            decoded_to_txt_input_ids=decoded_to_txt_input_ids,
            backbone=self.backbone,
            scanpath=scanpath,
        )

        # sp_pos is batch_data.scanpath, when adding 2 to each element that is not -1, add a 0 column at the beginning and add 1 to the wholte tensor
        sp_pos = scanpath.clone()
        sp_pos[sp_pos != -1] += 1
        sp_pos = torch.cat(
            (torch.zeros(sp_pos.shape[0], 1).to(sp_pos.device), sp_pos), dim=1
        )
        sp_pos += 1
        sp_pos = sp_pos.int()

        # unused_sp_ordinal_pos = batch_data.fixation_features[:, :, 0].int() #! TODO why not used? delete?

        sp_fix_dur = fixation_features[
            ..., 1
        ]  #! The feature order is hard coded in model_args. Make sure it's correct
        sp_landing_pos = fixation_features[..., 2]

        # add a column of zeros to both sp_fix_dur and sp_landing_pos to account for the <s> token
        sp_fix_dur = torch.cat(
            (torch.zeros(sp_fix_dur.shape[0], 1).to(sp_fix_dur.device), sp_fix_dur),
            dim=1,
        )
        sp_landing_pos = torch.cat(
            (
                torch.zeros(sp_landing_pos.shape[0], 1).to(sp_landing_pos.device),
                sp_landing_pos,
            ),
            dim=1,
        )

        sn_embd = batch_data.input_ids
        sn_mask = batch_data.input_masks
        # if the second dimension of the scanpath is more than the maximum context length (of self.bert_encoder), cut it and notify
        bert_encoder_max_len = self.bert_encoder.config.max_position_embeddings
        if sp_input_ids.shape[1] > bert_encoder_max_len - 1:
            print(
                f"Text length is more than the maximum context length of the model ({bert_encoder_max_len}). Cutting from the BEGINNING of the text to max length."
            )
            sn_embd = sn_embd[:, : bert_encoder_max_len - 1]
            sn_mask = sn_mask[:, : bert_encoder_max_len - 1]

        logits, x = self(
            sn_emd=sn_embd,
            sn_mask=sn_mask,
            word_ids_sn=word_ids_sn,
            sn_word_len=sn_word_len,
            sp_emd=sp_input_ids,
            sp_pos=sp_pos,
            sp_fix_dur=sp_fix_dur,
            sp_landing_pos=sp_landing_pos,
            word_ids_sp=word_ids_sp,
        )

        labels = batch_data.labels

        if (
            self.prediction_mode in (PredMode.CHOSEN_ANSWER, PredMode.QUESTION_LABEL)
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
            loss = self.loss(logits.squeeze(), labels)

        if self.model_args.add_contrastive_loss:
            if x.ndim == 1:
                x = x.unsqueeze(0)
            contrastive_loss = self.contrastive_loss(x, ordered_labels)
            loss += self.cl_alpha * contrastive_loss
        return ordered_labels, loss, ordered_logits.squeeze(), labels, logits.squeeze()

    def order_labels_logits(self, logits, labels, answer_mapping):
        # Get the sorted indices of answer_mapping along dimension 1
        sorted_indices = answer_mapping.argsort(dim=1)
        # Use these indices to rearrange each row in logits
        ordered_logits = torch.gather(logits, 1, sorted_indices)
        ordered_label = answer_mapping[range(answer_mapping.shape[0]), labels]

        return ordered_label, ordered_logits
