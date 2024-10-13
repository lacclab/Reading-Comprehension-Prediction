import ast
import json
import os

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset as TorchTensorDataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from src.configs.constants import DataRepresentation, Fields, PredMode
from src.configs.main_config import Args

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid warnings


class TextDataSet(TorchDataset):
    """
    A PyTorch dataset for text data.
    """

    def __init__(self, cfg: Args):
        self.prediction_mode = cfg.model.prediction_config.prediction_mode
        self.max_seq_len = cfg.model.max_seq_len
        self.actual_max_seq_len = 0
        self.prepend_eye_data = cfg.model.model_params.prepend_eye_data
        self.add_answers = cfg.model.prediction_config.add_answers
        self.preorder = cfg.model.preorder
        print(
            f"{self.prediction_mode=}, {self.prepend_eye_data=}, {self.add_answers=}, {self.preorder=}"
        )
        self.print_tokens = True
        usecols = [
            field.value
            for field in [
                Fields.BATCH,
                Fields.ARTICLE_ID,
                Fields.LEVEL,
                Fields.PARAGRAPH_ID,
                Fields.Q_IND,
                Fields.LIST,
                Fields.QUESTION,
                Fields.A,
                Fields.B,
                Fields.C,
                Fields.D,
                Fields.PARAGRAPH,
                Fields.CORRECT_ANSWER,
                Fields.ANSWERS_ORDER,
                Fields.PRACTICE,
                Fields.REREAD,
                Fields.HAS_PREVIEW,
            ]
        ]
        text_data = pd.read_csv(
            cfg.data_path.text_data_path,
            sep="\t",
            usecols=usecols,
        )

        text_data = text_data[text_data[Fields.PRACTICE] == 0]
        print("Removed practice texts")

        # REREAD is not needed as the texts are the same.
        text_data.drop(columns=[Fields.PRACTICE, Fields.REREAD], inplace=True)
        text_data.drop_duplicates(inplace=True)
        self.tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(
            cfg.model.backbone,
            add_special_tokens=False,
            is_split_into_words=True,
            add_prefix_space=True,
        )
        eye_token = "<eye>"
        self.tokenizer.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": [eye_token]},  # type: ignore
            replace_additional_special_tokens=False,
        )
        self.eye_token_id: int = self.tokenizer.convert_tokens_to_ids(eye_token)  # type: ignore

        self.text_key_fields = [
            Fields.BATCH,
            Fields.ARTICLE_ID,
            Fields.PARAGRAPH_ID,
            Fields.Q_IND,
            Fields.LEVEL,
            Fields.LIST,
            Fields.ANSWERS_ORDER,
            Fields.HAS_PREVIEW,
        ]  #! TODO Fields here MUST be present in the config.data_args.py groupby_columns
        text_data.drop_duplicates(subset=self.text_key_fields, inplace=True)
        text_data = text_data.reset_index(drop=True)

        text_keys = (
            text_data[self.text_key_fields].copy().astype(str).apply("_".join, axis=1)
        )

        # create a dict mapping from key column (as the dict key) to index (as the dict value)
        self.key_to_index = dict(zip(text_keys, text_keys.index))

        if self.prediction_mode in (
            PredMode.QUESTION_LABEL,
            PredMode.QUESTION_n_CONDITION,
        ):
            text_data = self.add_question_prediction_labels(text_data)
            text_data = self.add_other_questions_to_text(text_data)
            text_data = self.add_lonely_and_coupled_questions_data_to_text(text_data)

        (
            self.text_features,
            self.inversions_lists,
        ) = self.convert_examples_to_features(
            text_data, concat_or_duplicate=cfg.model.model_params.concat_or_duplicate
        )

        self.text_data = text_data

    def __len__(self):
        return len(self.key_to_index)

    def __getitem__(self, index: int):
        features = self.text_features[index]
        inversions_list = self.inversions_lists[index]
        return features, inversions_list

    def add_question_prediction_labels(self, text_data: pd.DataFrame) -> pd.DataFrame:
        print("Adding other questions to text data")

        with open(
            file="data/interim/onestop_qa.json",  # TODO Move to config
            mode="r",
            encoding="utf-8",
        ) as f:
            RAW_TEXT = json.load(f)

        def get_article_data(article_id: str) -> dict:
            for article in RAW_TEXT["data"]:
                if article["article_id"] == article_id:
                    return article
            raise ValueError(f"Article id {article_id} not found")

        question_prediction_labels = []
        for row in tqdm(
            iterable=text_data.itertuples(), total=len(text_data), desc="Adding"
        ):
            # Filter the original DataFrame for the current paragraph
            full_article_id = f"{row.batch}_{row.article_id}"
            questions = pd.DataFrame(
                get_article_data(full_article_id)["paragraphs"][row.paragraph_id - 1][
                    "qas"
                ]
            ).drop(["answers"], axis=1)
            assert len(questions) == 3, f"Expected 3 questions for paragraph \
            {row.paragraph_id}, got {len(questions)}"

            question_prediction_labels.append(
                questions.loc[
                    questions["q_ind"] == row.q_ind, "question_prediction_label"
                ].values[0]
            )

        text_data["question_prediction_label"] = question_prediction_labels

        return text_data

    def add_other_questions_to_text(self, text_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the text data for question prediction.
        Guarantees that the if another question exists that references the same question, is is
        stored as 'other_question_1. If another question exists that references the same question
        does not exist, the two other questions are stored as 'other_question_1' and
        'other_question_2' randomly.

        Args:
            text_data (pd.DataFrame): The text data to prepare.

        Returns:
            pd.DataFrame: The prepared text data.

        """
        print("Adding other questions to text data")

        with open(
            file="data/interim/onestop_qa.json",  # TODO Move to config
            mode="r",
            encoding="utf-8",
        ) as f:
            RAW_TEXT = json.load(f)

        def get_article_data(article_id: str) -> dict:
            for article in RAW_TEXT["data"]:
                if article["article_id"] == article_id:
                    return article
            raise ValueError(f"Article id {article_id} not found")

        other_questions_1, other_questions_2 = [], []
        for row in tqdm(
            iterable=text_data.itertuples(), total=len(text_data), desc="Adding"
        ):
            # Filter the original DataFrame for the current paragraph
            full_article_id = f"{row.batch}_{row.article_id}"
            questions = pd.DataFrame(
                get_article_data(full_article_id)["paragraphs"][row.paragraph_id - 1][
                    "qas"
                ]
            ).drop(["answers"], axis=1)
            assert len(questions) == 3, f"Expected 3 questions for paragraph \
            {row.paragraph_id}, got {len(questions)}"

            row_index = questions.loc[questions["q_ind"] == row.q_ind].index[0]
            reference_value: np.int64 = questions.at[row_index, "references"]

            other_question_2 = questions.loc[
                questions["references"] != reference_value, "question"
            ]
            if len(other_question_2) == 1:
                other_question_2 = other_question_2.item()

                other_question_1 = questions.loc[
                    (questions["references"] == reference_value)
                    & (questions["q_ind"] != row.q_ind),
                    "question",
                ].iloc[0]  # type: ignore
            elif len(other_question_2) == 2:
                # arbitrarily choose the order of the questions
                other_question_1 = other_question_2.values[0]
                other_question_2 = other_question_2.values[1]
            else:
                raise ValueError(
                    f"Expected 2 other questions, got {len(other_question_2)}"
                )

            # make sure that the other questions are not the same
            assert (
                other_question_1 != other_question_2
            ), f"Other questions are the same: {other_question_1}"
            # and that they are not the same as the reference question
            assert (
                other_question_1 != row.question
            ), f"Other question is the same as the reference question: {other_question_1}"
            assert (
                other_question_2 != row.question
            ), f"Other question is the same as the reference question: {other_question_2}"

            other_questions_1.append(other_question_1)
            other_questions_2.append(other_question_2)

        # Add the other questions to the DataFrame
        text_data["other_question_1"] = other_questions_1
        text_data["other_question_2"] = other_questions_2

        return text_data

    def add_lonely_and_coupled_questions_data_to_text(
        self, text_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This function adds the following columns to text data:
        - "lonely_question": The question that doesn't share critical span with another.
        - "couple_question_1": The first question in a pair of questions that share a critical span.
        - "couple_question_2": The second question in a pair of questions that share a critical span.
        """

        print("Adding other questions to text data")

        with open(
            file="data/interim/onestop_qa.json",  # TODO Move to config
            mode="r",
            encoding="utf-8",
        ) as f:
            RAW_TEXT = json.load(f)

        def get_article_data(article_id: str) -> dict:
            for article in RAW_TEXT["data"]:
                if article["article_id"] == article_id:
                    return article
            raise ValueError(f"Article id {article_id} not found")

        lonely_questions, couple_questions_1, couple_questions_2 = [], [], []
        for row in tqdm(
            iterable=text_data.itertuples(), total=len(text_data), desc="Adding"
        ):
            # Filter the original DataFrame for the current paragraph
            full_article_id = f"{row.batch}_{row.article_id}"
            questions = pd.DataFrame(
                get_article_data(full_article_id)["paragraphs"][row.paragraph_id - 1][
                    "qas"
                ]
            ).drop(["answers"], axis=1)
            assert len(questions) == 3, f"Expected 3 questions for paragraph \
            {row.paragraph_id}, got {len(questions)}"

            lonely_question = questions.loc[
                questions["question_prediction_label"] == 0, "question"
            ].item()
            couple_question_1 = questions.loc[
                questions["question_prediction_label"] == 1, "question"
            ].item()
            couple_question_2 = questions.loc[
                questions["question_prediction_label"] == 2, "question"
            ].item()

            # make sure that the other questions are not the same
            assert (
                couple_question_1 != couple_question_2
            ), f"Other questions are the same: {couple_question_1}"
            # note the any of the couple questions and the lonely question can be row.question

            lonely_questions.append(lonely_question)
            couple_questions_1.append(couple_question_1)
            couple_questions_2.append(couple_question_2)

        # Add the other questions to the DataFrame
        text_data["couple_question_1"] = couple_questions_1
        text_data["couple_question_2"] = couple_questions_2
        text_data["lonely_question"] = lonely_questions

        return text_data

    def convert_examples_to_features(
        self,
        examples: pd.DataFrame,
        concat_or_duplicate: DataRepresentation,
    ) -> tuple[TorchTensorDataset, list[list[int]]]:
        ### Roberta tokenization
        """Loads a data file into a list of `InputBatch`s."""

        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.cls_token_id is not None
        paragraphs_input_ids_list = []
        paragraphs_masks_list = []
        input_ids_list = []
        input_masks_list = []
        labels_list = []
        passages_length = []
        inversions_list = []
        questions_orders = []

        c = pd.read_csv("data/interim/consecutive_same_values_unique_2.csv")

        for example in tqdm(
            examples.itertuples(),
            total=len(examples),
            desc="Tokenizing",
        ):
            # TODO consider using merge instead of iterating through examples to find matches
            matches = c[
                (c["paragraph_id"] == example.paragraph_id)
                & (c["list"] == example.list)
                & (c["level"] == example.level)
                & (c["batch"] == example.batch)
                & (c["has_preview"] == example.has_preview)
                & (c["article_id"] == example.article_id)
            ]
            paragraph = example.paragraph
            if len(matches) > 0:
                for match_ in matches.itertuples():
                    paragraph = self.fix_paragraph(
                        paragraph,  # type: ignore
                        match_.IA_LABEL,  # type: ignore
                        match_.IA_ID,  # type: ignore
                    )

            paragraph_ids, inversions = self.tokenize(text=paragraph)  # type: ignore

            # TODO is this necessary? can be extracted from context_tokens or others instead?
            p_input_ids = paragraph_ids.copy()

            p_input_ids.insert(0, self.tokenizer.cls_token_id)
            p_input_mask = [1] * len(p_input_ids) + [0] * (
                self.max_seq_len - len(p_input_ids)
            )
            # Zero-pad up to the sequence length.
            p_padding_ids = [1] * (self.max_seq_len - len(p_input_ids))  # 1 for roberta
            p_input_ids += p_padding_ids
            paragraphs_input_ids_list.append(p_input_ids)
            paragraphs_masks_list.append(p_input_mask)

            (
                start_ending_ids,
                endings_ids,
                question_order,
            ) = self.prepare_endings_based_on_mode(example)

            if concat_or_duplicate == DataRepresentation.CONCAT:
                full_ending_ids = start_ending_ids if start_ending_ids else []
                # TODO make sure doesn't break other parts
                # if start_ending_ids:
                #     full_ending_ids.append(self.tokenizer.sep_token_id)
                #     full_ending_ids.append(self.tokenizer.sep_token_id)

                for ending_index, ending_tokens in enumerate(endings_ids):
                    # In Q prediction mode, there are no start_ending_tokens,
                    # so don't add before first ending.
                    full_ending_ids.extend(ending_tokens)
                    # TODO make sure doesn't break other parts
                    # if ending_index < len(endings_ids) - 1:
                    # full_ending_ids.append(self.tokenizer.sep_token_id)
                    # full_ending_ids.append(self.tokenizer.sep_token_id)

                input_ids, input_masks = self.process_example(
                    paragraph_ids, full_ending_ids
                )

            elif concat_or_duplicate == DataRepresentation.DUPLICATE:
                input_ids, input_masks = self.duplicate_example(
                    paragraph_ids, start_ending_ids, endings_ids
                )
            if self.print_tokens:
                if isinstance(input_ids[0], list):
                    for ids in input_ids:
                        print(self.tokenizer.convert_ids_to_tokens(ids))
                else:
                    print(self.tokenizer.convert_ids_to_tokens(input_ids))  # type: ignore
                self.print_tokens = False

            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)
            labels_list.append(example.correct_answer)
            passages_length.append(len(paragraph_ids))
            inversions_list.append(inversions)
            questions_orders.append(question_order)

        print(f"{self.max_seq_len=}. Max length in practice={self.actual_max_seq_len}.")

        features = TorchTensorDataset(
            torch.tensor(paragraphs_input_ids_list, dtype=torch.long),
            torch.tensor(paragraphs_masks_list, dtype=torch.long),
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(input_masks_list, dtype=torch.long),
            torch.tensor(labels_list, dtype=torch.long),
            torch.tensor(passages_length, dtype=torch.long),
            torch.tensor(questions_orders, dtype=torch.long),
        )

        return features, inversions_list

    def duplicate_example(
        self,
        context_ids: list[int],
        start_ending_ids: list[int] | None,
        endings_ids: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        choices_ids = []
        choices_masks = []
        sep_token_id = self.tokenizer.sep_token_id
        assert sep_token_id is not None
        for ending_ids in endings_ids:
            if start_ending_ids:  #  means we are predicting answer
                full_ending_ids = (
                    start_ending_ids
                    + [sep_token_id]
                    + [sep_token_id]
                    + ending_ids  # TODO delete sep sep?
                )
            else:
                full_ending_ids = (
                    ending_ids  # TODO for q prediction put here [] if want without q
                )
            input_ids, input_mask = self.process_example(context_ids, full_ending_ids)
            choices_ids.append(input_ids)
            choices_masks.append(input_mask)
        return choices_ids, choices_masks

    def build_inputs_with_special_tokens(
        self,
        context_ids: list[int],
        ending_ids: list[int],
    ) -> list[int]:
        """
        Based on from RobertaTokenizer.build_inputs_with_special_tokens
        #! Check where things break if making changes here
        """
        assert self.tokenizer.cls_token_id is not None
        assert self.tokenizer.sep_token_id is not None

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        input_ids = [cls_token_id]

        if self.prepend_eye_data:
            input_ids.extend([self.eye_token_id, sep_token_id])

        input_ids += (
            context_ids + [sep_token_id, sep_token_id] + ending_ids + [sep_token_id]
        )
        return input_ids

    def process_example(
        self,
        paragraph_ids: list[int],
        ending_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        input_ids = self.build_inputs_with_special_tokens(paragraph_ids, ending_ids)

        self.verify_input_length(input_ids)
        padding_length = self.max_seq_len - len(input_ids)
        # Update input mask and padding for the concatenated sequence
        input_mask = [1] * len(input_ids) + [0] * padding_length
        padding_ids = [1] * padding_length  # 1 for roberta
        input_ids.extend(padding_ids)

        return input_ids, input_mask

    def prepare_endings_based_on_mode(
        self, example
    ) -> tuple[list[int] | None, list[list[int]], list[int]]:
        """
        Process the example based on the prediction mode.

        Args:
            example: The data example to be processed.

        Returns:
            tuple[list[str], list[str], list[int]]:
                A tuple containing the start ending tokens, the endings, and the question order.
        """
        if self.prediction_mode in (
            PredMode.CHOSEN_ANSWER,
            PredMode.CORRECT_ANSWER,
            PredMode.IS_CORRECT,
            PredMode.CONDITION,
        ):
            question_order = []
            question = "Question: " + example.question
            if self.add_answers:
                start_ending_ids, _ = self.tokenize(text=question)
                endings = [example.a, example.b, example.c, example.d]
                # * Reorder endings based on answer_map
                if self.preorder:
                    answer_map = getattr(example, Fields.ANSWERS_ORDER)
                    answer_map = [
                        int(x) for x in ast.literal_eval(answer_map.replace(" ", ","))
                    ]
                    endings = [endings[i] for i in answer_map]
                    endings = [
                        f"Answers: (correct) {endings[0]}",
                        f"(wrong 1) {endings[1]}",
                        f"(wrong 2) {endings[2]}",
                        f"(wrong 3) {endings[3]}",
                    ]
                else:
                    endings = [
                        f"Answers: (answer 1) {endings[0]}",
                        f"(answer 2) {endings[1]}",
                        f"(answer 3) {endings[2]}",
                        f"(answer 4) {endings[3]}",
                    ]
            else:
                start_ending_ids = None
                endings = [question]
        elif self.prediction_mode in (
            PredMode.QUESTION_LABEL,
            PredMode.QUESTION_n_CONDITION,
        ):
            start_ending_ids = None
            endings = [
                example.lonely_question,
                example.couple_question_1,
                example.couple_question_2,
            ]
            question_order = [
                0,
                1,
                2,
            ]  # "lonely_question", "couple_question_1", "couple_question_2"
        else:
            raise ValueError(
                f"Invalid value for PREDICTION_MODE: {self.prediction_mode}"
            )
        """elif self.prediction_mode in (PredMode.QUESTION_LABEL, PredMode.QUESTION_n_CONDITION):
            start_ending_ids = None
            endings = [example.question]
            if example.q_ind == 0:
                endings += [example.other_question_1, example.other_question_2]
                question_order = [0, 1, 2]
            elif example.q_ind == 1:
                endings = (
                    [example.other_question_1] + endings + [example.other_question_2]
                )
                question_order = [1, 0, 2]
            elif example.q_ind == 2:
                endings = [example.other_question_1, example.other_question_2] + endings
                question_order = [1, 2, 0]
            else:
                raise ValueError("Invalid value for example.q_ind")"""  # old code

        endings_ids: list[list[int]] = [self.tokenize(ending)[0] for ending in endings]
        # Add empty list for the gathering condition
        if self.prediction_mode in (PredMode.QUESTION_n_CONDITION):
            endings_ids += [[]]
            question_order += [3]
        return start_ending_ids, endings_ids, question_order

    def verify_input_length(self, tokens: list[int]) -> None:
        assert (
            len(tokens) <= self.max_seq_len
        ), f"tokens length is {len(tokens)}, max_seq_length is {self.max_seq_len}"

        if len(tokens) > self.actual_max_seq_len:
            self.actual_max_seq_len = len(tokens)

    def tokenize(self, text: str) -> tuple[list[int], list[int]]:
        """
        Tokenizes a paragraph into a list of tokens.

        Args:
            paragraph (str): The paragraph to tokenize.

        Returns:
            tuple[list[str], list[int]]: The tokenized paragraph and the inversions list.

        """
        tokens = self.tokenizer(
            text.split(), is_split_into_words=True, add_special_tokens=False
        )
        input_ids: list[int] = tokens["input_ids"]  # type: ignore
        inversions: list[int] = tokens.word_ids()  # type: ignore

        return input_ids, inversions

    @staticmethod
    def fix_paragraph(paragraph: str, word: str, ia_id: int) -> str:
        # Split the paragraph into words
        words = paragraph.split()

        # Check if the word at the given index is the word to replace
        for i in range(ia_id - 3, ia_id + 1):
            if i < 0 or i >= len(words):
                continue
            if words[i] == word:
                # Replace the word
                if word == "6.30am;":  # TODO remove hardcoding
                    words[i] = "6.30 am;"
                words[i] = words[i].replace("-", "- ", 1)
                break

        # Join the words back into a paragraph
        new_paragraph = " ".join(words)

        return new_paragraph
