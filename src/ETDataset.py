"""This module contains the dataset classes for the project."""

import ast
import itertools
import pickle
import warnings
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Union, Callable

import numpy as np
import torch
import wandb
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset as TorchTensorDataset
from tqdm import tqdm

from src.configs.constants import (
    MAX_SCANPATH_LENGTH,
    NUM_ANSWERS,
    NUM_QUESTIONS,
    SCANPATH_PADDING_VAL,
    DataRepresentation,
    Fields,
    ItemLevelFeaturesModes,
    ModelNames,
    NormalizationModes,
    PredMode,
    groupby_mappings,
    gsf_features,
)
from src.configs.main_config import Args
from src.TextDataSet import TextDataSet

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd  # noqa: E402

try:
    import swifter  # noqa: F401
except ImportError:
    "swifter not installed. Install with pip install swifter[groupby]"


class ETDataset(TorchDataset):
    """
    A PyTorch dataset for eye movement features.

    Args:
        cfg (Args): The configuration object.
        ia_data (pd.DataFrame): The IA data.
        fixation_data (pd.DataFrame): The fixation data.
        text_data (TextDataSet): The text data.
        ia_scaler (Scaler): The scaler for the IA features.
        fixation_scaler (Scaler): The scaler for the fixation features.
        name (str, optional): The name of the dataset. Defaults to "".
    """

    def __init__(
        self,
        cfg: Args,
        ia_data: pd.DataFrame,
        fixation_data: Union[pd.DataFrame, None],
        text_data: TextDataSet,
        ia_scaler: Union[MinMaxScaler, RobustScaler, StandardScaler],
        fixation_scaler: Union[MinMaxScaler, RobustScaler, StandardScaler, None],
        trial_features_scaler: Union[MinMaxScaler, RobustScaler, StandardScaler, None],
        regime_name: str = "",
        set_name: str = "",
    ):
        self.ia_features = cfg.model.ia_features
        self.fixation_features = (
            cfg.model.fixation_features + cfg.model.ia_features_to_add_to_fixation_data
        )
        self.ia_categorical_features = cfg.model.ia_categorical_features

        self.target_column = cfg.model.prediction_config.target_column
        self.normalize = cfg.data.normalization_mode
        if self.normalize == NormalizationModes.NONE:
            print("Not normalizing features")
        else:
            print(f"Normalizing features using {self.normalize} normalization")
        self.prediction_mode = cfg.model.prediction_config.prediction_mode
        self.concat_or_duplicate = cfg.model.model_params.concat_or_duplicate
        self.add_beyelstm_features = cfg.model.add_beyelstm_features
        self.max_seq_len = cfg.model.max_seq_len
        self.max_eye_len = cfg.model.max_eye_len
        self.prepend_eye_data = cfg.model.model_params.prepend_eye_data
        self.item_level_features_mode = cfg.model.model_params.item_level_features_mode
        # Extract the relevant columns from the input data.
        trial_groupby_columns = cfg.data.groupby_columns
        print(f"{trial_groupby_columns=}")
        raw_ia_data = ia_data[
            list(set(trial_groupby_columns + self.ia_features))
        ].copy()

        if nan_cols := raw_ia_data.columns[raw_ia_data.isna().any()].tolist():
            print(f"There are columns with NaN values: {nan_cols}!!!")
        self.grouped_ia_data = raw_ia_data.groupby(trial_groupby_columns)

        self.model_name = cfg.model.model_params.model_name

        self.ia_scaler = self.fit_scaler_if_not_fitted(
            ia_scaler, raw_ia_data, self.ia_features
        )

        use_eyes_only = cfg.model.prediction_config.use_eyes_only
        assert isinstance(cfg.data.ia_query, str)
        self.condition = (
            "H" if "has_preview==1" in cfg.data.ia_query else "G"
        )  # TODO not robust

        self.features_identifier = (
            f"{self.condition}_"
            f"eyes-only={use_eyes_only}_"
            f"prepend-eyes={self.prepend_eye_data}_"
            f"norm={self.normalize}_"
            f"max-text={self.max_seq_len}_"
            f"max-eye={self.max_eye_len}_"
            f"{self.model_name}_"
            f"answers={cfg.model.prediction_config.add_answers}_"
            f"trial-level={cfg.model.add_beyelstm_features}_"
            f"fixation={cfg.model.use_fixation_report}_"
            f"pred-mode={self.prediction_mode}_"
            f"{self.concat_or_duplicate}_"
            f"preorder={cfg.model.preorder}"
        )  # TODO changing the query or possibly other features could break this

        if len(self.ia_features) != 33:
            self.features_identifier += f"_ia-feats={len(self.ia_features)}"

        self.trial_level_features_identifier = (
            f"{self.condition}_"
            + f"_item-level-features-mode={self.item_level_features_mode}"
        )

        if (
            fixation_data is not None
            and fixation_scaler is not None
            and trial_features_scaler is not None
        ):
            raw_fixation_scanpath_ia_labels = fixation_data[
                trial_groupby_columns + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
            ]
            raw_fixation_data = fixation_data[
                list(set(trial_groupby_columns + self.fixation_features))
            ].copy()

            if cfg.model.add_beyelstm_features:
                raw_fixation_data = add_missing_features(
                    raw_fixation_data, trial_groupby_columns
                )

                trial_level_features_dir = Path(
                    "data/interim/trial_level_features/"
                    + self.trial_level_features_identifier
                    + f"/fold={cfg.data.fold_index}/"
                )
                trial_level_features_filename = f"{regime_name}_{set_name}.pkl"
                self.trial_level_features = self.cache_or_load_feature(
                    cache_dir_path=trial_level_features_dir,
                    cache_filename=trial_level_features_filename,
                    overwrite_feature=cfg.data.overwrite_data,
                    create_feature_func=self.compute_trial_level_features_parallel,
                    create_feature_func_args=dict(
                        raw_fixation_data=raw_fixation_data,
                        raw_ia_data=raw_ia_data,
                        trial_groupby_columns=trial_groupby_columns,
                    ),
                )

                self.trial_level_scaler = self.fit_scaler_if_not_fitted(
                    scaler=trial_features_scaler,
                    raw_data=self.trial_level_features,
                )

            self.grouped_raw_fixation_scanpath_ia_labels = (
                raw_fixation_scanpath_ia_labels.groupby(trial_groupby_columns)
            )

            self.fixation_scaler = self.fit_scaler_if_not_fitted(
                fixation_scaler, raw_fixation_data, self.fixation_features
            )

            self.grouped_fixation_data = raw_fixation_data.groupby(
                trial_groupby_columns
            )
        else:
            self.grouped_fixation_data = None

        self.ordered_key_list = list(self.grouped_ia_data.groups)
        labels = [
            key_[-1]
            for key_ in self.ordered_key_list  # type: ignore
        ]  # Assumes that the last element in the key is the label
        self.abcd_labels = [
            key_[-1]  # type: ignore
            for key_ in self.ordered_key_list  # TODO changed to -1 for IsCorrect, was -2 for abcd_answer_column!
        ]  # Save for sampler # TODO what is -2 (abcd_answer_column) and -1 (label)? Mention here!!!
        unordered_label_counts = self.organize_label_counts(
            labels, label_names=regime_name
        )
        if self.prediction_mode in (PredMode.CHOSEN_ANSWER,):
            abcd_answer = ia_data.drop_duplicates(
                subset=trial_groupby_columns
            ).abcd_answer  # TODO use self.abcd_labels instead?
            self.ordered_label_counts = self.organize_label_counts(
                abcd_answer, label_names=regime_name
            )
        elif self.prediction_mode == PredMode.CORRECT_ANSWER:
            # TODO Not tested
            correct_answer = ia_data.drop_duplicates(
                subset=trial_groupby_columns
            ).correct_answer
            self.ordered_label_counts = self.organize_label_counts(
                correct_answer, label_names=regime_name
            )
        elif self.prediction_mode in (
            PredMode.QUESTION_LABEL,
            PredMode.QUESTION_n_CONDITION,
        ):
            self.ordered_label_counts = unordered_label_counts  # TODO order?
        elif self.prediction_mode in (
            PredMode.CONDITION,
            PredMode.IS_CORRECT,
        ):
            self.ordered_label_counts = unordered_label_counts
        else:
            raise ValueError(
                f"Invalid value for PREDICTION_MODE: {self.prediction_mode}"
            )

        print(f"{regime_name=} {set_name=}")
        print(f"Unordered label counts:\n{unordered_label_counts}")
        print(f"Ordered label counts:\n{self.ordered_label_counts}")
        if wandb.run:
            wandb.run.log(
                {
                    f"unordered_label_counts/{regime_name}": wandb.Table(
                        dataframe=unordered_label_counts
                    ),
                    f"ordered_label_counts/{regime_name}": wandb.Table(
                        dataframe=self.ordered_label_counts
                    ),
                }
            )

        self.n_duplicates = self.get_n_duplicates()

        self.actual_max_tokens_in_word = 0
        self.max_tokens_in_word = 9  # TODO push up to config
        self.max_q_len = 100  # TODO push up to config and TODO reduce more
        self.print_first_nan_occurrences = True

        features_dir = Path(
            "data/interim/features/"
            + self.features_identifier
            + f"/fold={cfg.data.fold_index}/"
        )
        features_filename = f"{regime_name}_{set_name}.pkl"
        self.features = self.cache_or_load_feature(
            cache_dir_path=features_dir,
            cache_filename=features_filename,
            overwrite_feature=cfg.data.overwrite_data,
            create_feature_func=self.convert_examples_to_features,
            create_feature_func_args=dict(
                text_data=text_data, use_eyes_only=use_eyes_only
            ),
        )

        self.n_tokens = len(text_data.tokenizer)
        self.eye_token_id = text_data.eye_token_id
        self.sep_token_id = text_data.tokenizer.sep_token_id
        print(f"{self.actual_max_tokens_in_word=} but using {self.max_tokens_in_word=}")

    @staticmethod
    def cache_or_load_feature(
        cache_dir_path: Path,
        cache_filename: str,
        overwrite_feature: bool,
        create_feature_func: Callable,
        create_feature_func_args: dict,
    ):
        cache_file_path = cache_dir_path / cache_filename
        if overwrite_feature or not cache_file_path.exists():
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Caching features to {cache_file_path}")
            feature = create_feature_func(**create_feature_func_args)
            with open(cache_file_path, "wb") as f:
                pickle.dump(
                    feature, f
                )  # TODO can be problematic if running on multiple GPUs and saving the same file
        else:
            print(f"Loading features from {cache_file_path}")
            with open(cache_file_path, "rb") as f:
                feature = pickle.load(f)
        return feature

    def fit_scaler_if_not_fitted(
        self,
        scaler: Union[MinMaxScaler, RobustScaler, StandardScaler],
        raw_data: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
    ) -> Union[MinMaxScaler, RobustScaler, StandardScaler]:
        try:
            check_is_fitted(scaler)
        except NotFittedError:
            if not feature_columns:
                feature_columns = raw_data.columns.to_list()
            non_numeric = raw_data[feature_columns].select_dtypes(
                exclude=["number", "bool"]
            )
            if not non_numeric.empty:
                warnings.warn(
                    f"{non_numeric.columns.tolist()} are non-numeric and thus not normalized."
                )
            numeric_data = raw_data[feature_columns].drop(
                columns=self.ia_categorical_features, errors="ignore"
            )
            scaler.fit(numeric_data)
            print(f"Fitted {scaler} on {numeric_data.columns}")
        return scaler

    def get_n_duplicates(self) -> Literal[1, 3, 4]:
        if self.prediction_mode in (
            PredMode.CHOSEN_ANSWER,
            PredMode.CORRECT_ANSWER,
        ):
            if self.concat_or_duplicate == DataRepresentation.CONCAT:
                n_duplicates = 1
            elif self.concat_or_duplicate == DataRepresentation.DUPLICATE:
                n_duplicates = NUM_ANSWERS  # TODO consider replacing with len(self.class_names) or similar
            else:
                raise ValueError(
                    f"Invalid value for CONCAT_OR_DUPLICATE: {self.concat_or_duplicate}"
                )

        elif self.prediction_mode == PredMode.QUESTION_LABEL:
            if self.concat_or_duplicate == DataRepresentation.CONCAT:
                n_duplicates = 1
            elif self.concat_or_duplicate == DataRepresentation.DUPLICATE:
                n_duplicates = NUM_QUESTIONS
            else:
                raise ValueError(
                    f"Invalid value for CONCAT_OR_DUPLICATE: {self.concat_or_duplicate}"
                )
        elif self.prediction_mode == PredMode.QUESTION_n_CONDITION:
            if self.concat_or_duplicate == DataRepresentation.CONCAT:
                n_duplicates = 1
            elif self.concat_or_duplicate == DataRepresentation.DUPLICATE:
                n_duplicates = NUM_QUESTIONS + 1  # +1 for the null question
            else:
                raise ValueError(
                    f"Invalid value for CONCAT_OR_DUPLICATE: {self.concat_or_duplicate}"
                )

        elif self.prediction_mode in (
            PredMode.CONDITION,
            PredMode.IS_CORRECT,
        ):
            n_duplicates = 1
        else:
            raise ValueError(
                f"Invalid value for PREDICTION_MODE: {self.prediction_mode}"
            )
        print(
            f"Prediction Mode: {self.prediction_mode}, \
              Concat or Duplicate: {self.concat_or_duplicate}, \
              Number of Duplicates: {n_duplicates}"
        )
        return n_duplicates

    @staticmethod
    def organize_label_counts(labels, label_names: str) -> pd.DataFrame:
        label_counts = np.unique(labels, return_counts=True)
        label_counts = pd.DataFrame(label_counts, index=["label", "count"]).T
        label_counts["percent"] = (
            label_counts["count"] / label_counts["count"].sum() * 100
        )

        label_counts["percent"] = label_counts["percent"].astype(float).round(2)
        label_counts.attrs["name"] = label_names
        return label_counts

    def __len__(self):
        """
        Get the number of unique groups in the dataset.

        Returns:
            The number of unique groups in the dataset.
        """
        return len(self.grouped_ia_data.groups)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_
        """
        return self.features[idx]

    def convert_examples_to_fixation_features(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the examples in the dataset to features.

        Returns:
            A list of tuples containing:
                - y (int): The target value for the item.
                - X (tensor): A PyTorch tensor containing the feature values for the item.
        """
        fixation_list = []
        pads_list = []
        scanpath_list = []
        scanpath_pads_list = []
        trial_level_features_list = []
        for idx in tqdm(range(len(self))):
            # Get the data group associated with the given index.
            grouped_data_key = self.ordered_key_list[idx]
            trial = self.grouped_fixation_data.get_group(grouped_data_key).copy()  # type: ignore

            scanpath = self.grouped_raw_fixation_scanpath_ia_labels[
                Fields.FIXATION_REPORT_IA_ID_COL_NAME
            ].get_group(grouped_data_key)

            fixation = trial[self.fixation_features].drop(
                columns=self.ia_categorical_features, errors="ignore"
            )

            if self.print_first_nan_occurrences:
                if fixation.columns[fixation.isna().any()].tolist():
                    warnings.warn(
                        f"{fixation.columns[fixation.isna().any()].tolist()}. Ffiling and bfilling."
                    )
                self.print_first_nan_occurrences = False
            fixation = fixation.ffill().bfill()

            if self.normalize:
                fixation = self.normalize_features(
                    fixation, normalize=self.normalize, mode="fixation"
                )

            if self.add_beyelstm_features:
                trial_features = self.trial_level_features.loc[grouped_data_key]  # type: ignore
                if self.normalize:
                    trial_features = self.normalize_features(
                        trial_features, normalize=self.normalize, mode="trial_level"
                    )
                trial_level_features_list.append(trial_features)
                # concat back the "Is_Content_Word" and "Reduced_POS" columns from trial
                fixation = np.concatenate(
                    (
                        fixation,
                        trial[["Is_Content_Word", "Reduced_POS"]].to_numpy(),
                    ),  #! Order matters here!
                    axis=1,
                )

            pad_length = MAX_SCANPATH_LENGTH - len(fixation)
            fixation_dim = fixation.shape[1]

            fixation_padding = np.zeros((pad_length, fixation_dim))
            fixation = np.concatenate((fixation, fixation_padding))
            # pad the scanpath with -1
            scanpath_padding = np.full(pad_length, SCANPATH_PADDING_VAL)
            scanpath = np.concatenate((scanpath, scanpath_padding))

            if self.n_duplicates > 1:
                fixation = np.repeat(fixation[np.newaxis, :], self.n_duplicates, axis=0)
                scanpath = np.repeat(scanpath[np.newaxis, :], self.n_duplicates, axis=0)

            fixation_list.append(fixation)
            pads_list.append(pad_length)
            scanpath_list.append(scanpath)
            scanpath_pads_list.append(pad_length)

        trial_level_features = (
            torch.tensor(np.array(trial_level_features_list), dtype=torch.float32)
            if trial_level_features_list
            else torch.empty((len(self),))
        )
        return (
            torch.tensor(np.array(fixation_list), dtype=torch.float32),
            torch.tensor(pads_list, dtype=torch.long),
            torch.tensor(np.array(scanpath_list), dtype=torch.long),
            torch.tensor(scanpath_pads_list, dtype=torch.long),
            trial_level_features,
        )

    def determine_answer_map(self, trial_info, questions_order):
        if self.prediction_mode in (
            PredMode.CHOSEN_ANSWER,
            PredMode.CORRECT_ANSWER,
        ):
            answer_map = trial_info[Fields.ANSWERS_ORDER]
            answer_map = [int(x) for x in ast.literal_eval(answer_map)]

        elif self.prediction_mode in (
            PredMode.QUESTION_LABEL,
            PredMode.QUESTION_n_CONDITION,
        ):
            answer_map = questions_order.tolist()

        elif self.prediction_mode in (
            PredMode.CONDITION,
            PredMode.IS_CORRECT,
        ):
            answer_map = []
        else:
            raise ValueError(
                f"Invalid value for PREDICTION_MODE: {self.prediction_mode}"
            )
        return answer_map

    def group_to_length(
        self,
        lst: list[int],
        col_pad_to_len: int,
        row_pad_to_len: int,
        inv_list_to_token_word_attn_mask: bool = False,
    ) -> torch.Tensor:
        """
        Pad a list of values to a predefined length.

        Example: [1, 1, 1, 2, 2, 3, 3, 3, 3] -> [tensor([[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, 7, 8]])]
        Three words, first word has 3 tokens, second word has 2 tokens, third word has 4 tokens.
        Input list assumed to be sorted.
        Used to represent token to word mapping.
        I.e., in input, each token (index in lst) is mapped to a word index (value in lst),
        in output each word index (row) is mapped to a token index (values in row).

        Args:
            lst (list): The list of values to pad.
            col_pad_to_len (int): The length to pad to number of cols.
            row_pad_to_len (int): The length to pad to number of rows.

        Returns:
            A tensor containing the padded values.



        """
        # Group the list by the values, and convert to a tensor
        # Example: [1, 1, 1, 2, 2, 3, 3, 3, 3] -> [tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6, 7, 8])
        grouped_lst = [
            torch.tensor(data=list(group))
            for _, group in itertools.groupby(
                iterable=range(len(lst)), key=lambda x: lst[x]
            )
        ]

        if inv_list_to_token_word_attn_mask:
            """[1, 1, 1, 2, 2, 3, 3, 3, 3] -> [tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6, 7, 8])
                Before attending previous and next word:
                [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                 ...
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

                 After attending previous and next word:
                [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                 ...
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            Note: the first column is used to attend to the [CLS] token.
            """
            size = self.max_seq_len  # TODO: Can be reduced to maximal num of *words* in a paragraph (not tokens)
            matrix = torch.zeros(size + 1, size)
            for i, row in enumerate(grouped_lst):
                matrix[i, 0] = 1
                matrix[i, row + 1] = 1
                # Add attention to the previous and next word
                if i > 0:
                    matrix[i, grouped_lst[i - 1] + 1] = 1
                if i < len(grouped_lst) - 1:
                    matrix[i, grouped_lst[i + 1] + 1] = 1
            return matrix

        current_max_tokens_in_word = max(len(group) for group in grouped_lst)
        if current_max_tokens_in_word > self.actual_max_tokens_in_word:
            self.actual_max_tokens_in_word = current_max_tokens_in_word

        # Add padding
        padded_tensor = pad_sequence(
            sequences=grouped_lst, batch_first=True, padding_value=-2
        )
        # Example: [tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6, 7, 8])] -> [tensor([[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, 7, 8]])]

        num_padding_cols = max(0, col_pad_to_len - padded_tensor.size(dim=1))
        padding = torch.full(
            size=(padded_tensor.size(dim=0), num_padding_cols), fill_value=-2
        )
        padded_tensor = torch.cat(tensors=(padded_tensor, padding), dim=1)

        # Calculate the number of rows needed to reach the predefined length
        num_padding_rows = max(0, row_pad_to_len - padded_tensor.size(dim=0))

        # Create a tensor of padding values
        padding = torch.full(
            size=(num_padding_rows, padded_tensor.size(dim=1)), fill_value=-2
        )

        # Concatenate the padding to the padded_tensor
        padded_tensor = torch.cat(tensors=(padded_tensor, padding), dim=0)
        padded_tensor += 1
        return padded_tensor

    def convert_examples_to_features(
        self, text_data: TextDataSet, use_eyes_only: bool
    ) -> TorchDataset:
        """
        Convert the examples in the dataset to features.

        Returns:
            A list of tuples containing:
                - y (int): The target value for the item.
                - X (tensor): A PyTorch tensor containing the feature values for the item.
        """

        if self.grouped_fixation_data is not None:
            (
                fixation_tensor,
                fixation_pads_tensor,
                scanpath_tensor,
                scanpath_pads_tensor,
                trial_level_features_tensor,
            ) = self.convert_examples_to_fixation_features()
        else:
            fixation_tensor = torch.empty((len(self),))
            fixation_pads_tensor = torch.empty((len(self),))
            scanpath_tensor = torch.empty((len(self),))
            scanpath_pads_tensor = torch.empty((len(self),))
            trial_level_features_tensor = torch.empty((len(self),))

        input_ids_list = []
        input_masks_list = []
        labels_list = []
        eyes_list = []
        answer_mappings = []
        p_input_ids_list = []
        p_input_masks_list = []
        inversions_lists = []
        inversions_lists_pads = []
        grouped_inversions = []
        question_ids_list = []
        question_masks_list = []
        actual_max_q_length = 0
        for idx in tqdm(range(len(self)), desc="Converting"):
            # Get the data group associated with the given index.
            grouped_data_key = self.ordered_key_list[idx]
            trial = self.grouped_ia_data.get_group(grouped_data_key)
            # take trial info from first row
            trial_info = trial.iloc[0]
            key_ = trial_info[text_data.text_key_fields].astype(str).str.cat(sep="_")
            key_ = key_.replace("'", "").replace(",", "")
            text_index = text_data.key_to_index[key_]
            (
                (
                    p_input_ids,
                    p_input_masks,
                    input_ids,
                    input_mask,
                    unused_correct_answer,
                    passage_length,
                    questions_order,
                ),
                inversions_list,
            ) = text_data[text_index]

            input_ids_list.append(input_ids)

            # add dimension if only one question
            input_ids_unsqueeze = input_ids
            input_mask_unsqueeze = input_mask
            if len(input_ids.shape) == 1:
                input_ids_unsqueeze = input_ids_unsqueeze.unsqueeze(dim=0)
                input_mask_unsqueeze = input_mask_unsqueeze.unsqueeze(dim=0)

            additional_tokens = 5 if self.prepend_eye_data else 3

            question_mask = input_mask_unsqueeze[
                :, passage_length + additional_tokens :
            ]
            question_ids = input_ids_unsqueeze[:, passage_length + additional_tokens :]

            # Assuming question_ids and question_mask are 2D tensors with shape (n_questions, seq_len)
            question_ids_padded = []
            question_masks_padded = []
            for i in range(self.n_duplicates):  # Process each question separately
                question_id = question_ids[i]
                question_m = question_mask[i]

                question_id = question_id[question_m != 0]
                question_m = question_m[question_m != 0]

                pad_length = self.max_q_len - len(question_id)
                if actual_max_q_length < len(question_id):
                    actual_max_q_length = len(question_id)

                question_id = torch.cat(
                    (
                        input_ids_unsqueeze[i, 0].unsqueeze(dim=0),
                        question_id,
                        torch.ones(pad_length, dtype=torch.long),
                    )
                )  # add [CLS] token at the beginning

                question_m = torch.cat(
                    (
                        input_mask_unsqueeze[i, 0].unsqueeze(dim=0),
                        question_m,
                        torch.zeros(pad_length, dtype=torch.long),
                    )
                )  # mask of [CLS] at the beginning

                # Replace the original question_ids and question_mask with the processed ones
                question_ids_padded.append(question_id)
                question_masks_padded.append(question_m)

            question_ids_list.append(question_ids_padded)
            question_masks_list.append(question_masks_padded)

            inversion_list_pads = self.max_seq_len - len(inversions_list)
            padded_inversions_list = inversions_list + [0] * inversion_list_pads
            inversions_lists.append(padded_inversions_list)
            inversions_lists_pads.append(inversion_list_pads)
            p_input_ids_list.append(p_input_ids)
            p_input_masks_list.append(p_input_masks)

            eyes = trial[self.ia_features].drop(
                columns=self.ia_categorical_features, errors="ignore"
            )

            if self.normalize:
                eyes = self.normalize_features(
                    eyes, normalize=self.normalize, mode="ia"
                )
            else:
                eyes = eyes.to_numpy()
            assert eyes is not None

            if self.prepend_eye_data:
                num_pre_eye_tokens = 0
            else:
                aligned_eyes = [eyes[inv_idx, :] for inv_idx in inversions_list]
                assert passage_length == len(aligned_eyes)
                eyes = np.stack(aligned_eyes)

                #  TODO left pad can be removed after we verify that it doesn't break anything.
                num_pre_eye_tokens = 1

            # Pad zero vectors for eyes vectors to account for [CLS] / [SEP] tokens TODO doc
            eye_seq_len, eyes_dim = eyes.shape
            eyes_pad_left = np.zeros((num_pre_eye_tokens, eyes_dim))

            # pad_length = max_seq_length - passage_length - 1  # -1 for [CLS] token TODO make use
            pad_length = self.max_eye_len - eye_seq_len - num_pre_eye_tokens
            eyes_pad_right = np.zeros((pad_length, eyes_dim))
            eyes = np.concatenate((eyes_pad_left, eyes, eyes_pad_right))

            inv_list_to_token_word_attn_mask = (
                self.model_name == ModelNames.POSTFUSION_MODEL
            )

            group_inversions = self.group_to_length(
                lst=inversions_list,
                col_pad_to_len=self.max_tokens_in_word,
                row_pad_to_len=self.max_eye_len,
                inv_list_to_token_word_attn_mask=inv_list_to_token_word_attn_mask,
            )
            if self.grouped_fixation_data is not None:
                scanpath = (
                    scanpath_tensor[idx, 0, :]
                    if self.n_duplicates > 1
                    else scanpath_tensor[idx, :]
                )
                group_inversions = group_inversions[scanpath]
            if self.n_duplicates > 1:
                eyes = np.repeat(eyes[np.newaxis, :], self.n_duplicates, axis=0)
                group_inversions = np.repeat(
                    group_inversions[np.newaxis, :], self.n_duplicates, axis=0
                )
            eyes_list.append(eyes)

            if self.prepend_eye_data:
                if self.grouped_fixation_data is not None:
                    pad_len = fixation_pads_tensor[idx]
                    seq_len = MAX_SCANPATH_LENGTH - pad_len

                else:
                    seq_len = eye_seq_len
                    pad_len = pad_length

                ones = np.ones((seq_len))
                zeroes = np.zeros((pad_len))
                eye_mask = np.concatenate((ones, zeroes), axis=0)
                if self.n_duplicates > 1:
                    eye_mask = np.repeat(
                        eye_mask[np.newaxis, :], self.n_duplicates, axis=0
                    )
                    axis = 1
                else:
                    axis = 0
                # Prepend the eye_mask to input_mask as it is part of the input
                input_mask = torch.from_numpy(
                    np.concatenate((eye_mask, input_mask), axis=axis)
                )

            input_masks_list.append(input_mask)

            y = trial_info[self.target_column]
            labels_list.append(y)

            answer_map = self.determine_answer_map(
                trial_info=trial_info, questions_order=questions_order
            )
            answer_mappings.append(answer_map)

            grouped_inversions.append(group_inversions)

        input_ids_tensor = torch.stack(input_ids_list)
        input_masks_tensor = torch.stack(input_masks_list)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        eyes_tensor = torch.tensor(np.array(eyes_list), dtype=torch.float32)
        answer_mappings_tensor = torch.tensor(answer_mappings, dtype=torch.long)
        grouped_inversions_tensor = torch.stack(grouped_inversions)
        question_ids_tensor = torch.tensor(
            np.array(question_ids_list), dtype=torch.long
        )
        question_masks_tensor = torch.tensor(
            np.array(question_masks_list), dtype=torch.long
        )

        print(f"{actual_max_q_length=}; {self.max_q_len=}")

        # TODO consider replacing with BatchData from base_data directly to avoid this
        # TODO also consider computing these anyways all the time and just not use. (as is done now practically)
        if use_eyes_only:
            return (
                TorchTensorDataset(
                    eyes_tensor,
                    labels_tensor,
                    fixation_tensor,
                    fixation_pads_tensor,
                    scanpath_tensor,
                    scanpath_pads_tensor,
                    trial_level_features_tensor,
                )
                if self.grouped_fixation_data is not None
                else TorchTensorDataset(
                    eyes_tensor,
                    labels_tensor,
                )
            )
        elif self.grouped_fixation_data is not None:
            return TorchTensorDataset(
                torch.stack(p_input_ids_list),  # not in use by several models
                torch.stack(p_input_masks_list),  # not in use by several models
                input_ids_tensor,
                input_masks_tensor,
                labels_tensor,
                eyes_tensor,
                answer_mappings_tensor,
                fixation_tensor,
                fixation_pads_tensor,  # not in use by several models
                scanpath_tensor,  # not in use by several models
                scanpath_pads_tensor,
                torch.tensor(
                    inversions_lists, dtype=torch.long
                ),  # not in use by several models
                torch.tensor(
                    inversions_lists_pads, dtype=torch.long
                ),  # not in use by several models
                grouped_inversions_tensor,
                trial_level_features_tensor,  # not in use by several models
            )

        else:
            return TorchTensorDataset(
                input_ids_tensor,
                input_masks_tensor,
                labels_tensor,
                eyes_tensor,
                answer_mappings_tensor,
                grouped_inversions_tensor,
                question_ids_tensor,
                question_masks_tensor,
                torch.stack(p_input_ids_list),  # not in use by several models
                torch.stack(p_input_masks_list),  # not in use by several models
            )

    def normalize_features(
        self,
        x: pd.DataFrame | pd.Series,
        normalize: NormalizationModes,
        mode: Literal["ia", "fixation", "trial_level"],
    ) -> np.ndarray:
        """
        Z-score normalization of features based on the data/trial statistics.

        Parameters:
        X (pd.DataFrame): The feature matrix to normalize.
        normalize (str): The type of normalization to apply.
        Can be "all" to normalize based on the mean and standard deviation of the entire dataset,
        or "trial" to normalize based on the mean and standard deviation of the current trial.
        mean (np.ndarray): The mean values to use for normalization.
        std (np.ndarray): The standard deviation values to use for normalization.

        Returns:
        pd.DataFrame: The normalized feature matrix.
        """
        x = x.drop(columns=self.ia_categorical_features, errors="ignore")
        if mode == "ia":
            scaler = self.ia_scaler
        elif mode == "fixation":
            scaler = self.fixation_scaler
        elif mode == "trial_level":
            scaler = self.trial_level_scaler
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if normalize == NormalizationModes.NONE:
            return x.to_numpy()
        x_input = x.to_numpy().reshape(1, -1) if isinstance(x, pd.Series) else x
        if normalize == NormalizationModes.ALL:
            normalized_x = scaler.transform(x_input)
        elif normalize == NormalizationModes.TRIAL:
            normalized_x = scaler.fit_transform(x_input)
        else:
            raise ValueError(f"Invalid value for normalize: {normalize}")
        return normalized_x  # type: ignore

    @staticmethod
    def add_missing_categories_and_flatten(
        grouped_gsf_features: pd.DataFrame,
        groupby_fields: list,
        groupby_type_: str,
    ) -> pd.Series:
        new_index = grouped_gsf_features.index.union(
            pd.Index(groupby_fields)
        ).drop_duplicates()
        if len(groupby_fields) < len(new_index):
            print(
                f"Missing categories: {new_index.difference(groupby_fields)} in {groupby_type_}!"
            )
        grouped_gsf_features = grouped_gsf_features.reindex(new_index, fill_value=0)
        grouped_df_reset = grouped_gsf_features.reset_index()

        melted_ = grouped_df_reset.melt(
            id_vars=grouped_df_reset.columns[0],  # Use the first column as the id_vars
            var_name="variable",  # Name of the new variable column
            value_name="value",  # Name of the new value column
        )
        # If you want to add the groupby_type_ to the feature name so have feature names
        # melted_["feature_name"] = (
        #     groupby_type_
        #     + "_"
        #     + melted_['index'].astype(str)
        #     + "_"
        #     + melted_["variable"]
        # )
        # return melted_[["feature_name", "value"]].set_index("feature_name").sort_index()
        return melted_.sort_values(by="variable").value

    @staticmethod
    def get_gaze_entropy_features(
        x_means,
        y_means,
        x_dim=2560,
        y_dim=1440,
        patch_size=138,  # ±193 patches
    ) -> dict[str, int | float | np.float64]:
        # Gaze entropy measures detect alcohol-induced driver impairment - ScienceDirect
        # https://www.sciencedirect.com/science/article/abs/pii/S0376871619302789
        # computes the gaze entropy features
        # params:
        #    x_means: x-coordinates of fixations
        #    y_means: y coordinates of fixations
        #    x_dim: screen horizontal pixels
        #    y_dim: screen vertical pixels
        #    patch_size: size of patches to use
        # Based on https://github.com/aeye-lab/etra-reading-comprehension
        def calc_patch(patch_size, mean):
            return int(np.floor(mean / patch_size))

        def entropy(value):
            return value * (np.log(value) / np.log(2))

        # dictionary of visited patches
        patch_dict = {}
        # dictionary for patch transitions
        trans_dict = {}
        pre = None
        for i in range(len(x_means)):
            x_mean = x_means[i]
            y_mean = y_means[i]
            patch_x = calc_patch(patch_size, x_mean)
            patch_y = calc_patch(patch_size, y_mean)
            cur_point = f"{str(patch_x)}_{str(patch_y)}"
            if cur_point not in patch_dict:
                patch_dict[cur_point] = 0
            patch_dict[cur_point] += 1
            if pre is not None:
                if pre not in trans_dict:
                    trans_dict[pre] = []
                trans_dict[pre].append(cur_point)
            pre = cur_point

        # stationary gaze entropy
        # SGE
        sge = 0
        x_max = int(x_dim / patch_size)
        y_max = int(y_dim / patch_size)
        fix_number = len(x_means)
        for i in range(x_max):
            for j in range(y_max):
                cur_point = f"{str(i)}_{str(j)}"
                if cur_point in patch_dict:
                    cur_prop = patch_dict[cur_point] / fix_number
                    sge += entropy(cur_prop)
        sge = sge * -1

        # gaze transition entropy
        # GTE
        gte = 0
        for patch in trans_dict:
            cur_patch_prop = patch_dict[patch] / fix_number
            cur_destination_list = trans_dict[patch]
            (values, counts) = np.unique(cur_destination_list, return_counts=True)
            inner_sum = 0
            for i in range(len(values)):
                cur_count = counts[i]
                cur_prob = cur_count / np.sum(counts)
                cur_entropy = entropy(cur_prob)
                inner_sum += cur_entropy
            gte += cur_patch_prop * inner_sum
        gte = gte * -1
        return {"fixation_feature_SGE": sge, "fixation_feature_GTE": gte}

    @staticmethod
    def compute_fixation_trial_level_features(
        trial: pd.DataFrame,
        mode: ItemLevelFeaturesModes = ItemLevelFeaturesModes.DAVID,
    ) -> pd.Series:
        trial_level_features = []
        if mode == ItemLevelFeaturesModes.DAVID:
            gaze_features = ETDataset.get_gaze_entropy_features(
                x_means=trial["CURRENT_FIX_X"].values,
                y_means=trial["CURRENT_FIX_Y"].values,
            )
            trial_level_features.extend(gaze_features.values())

            total_num_fixations = len(trial)
            trial_level_features.append(total_num_fixations)

            total_num_words = trial["TRIAL_IA_COUNT"].drop_duplicates().values[0]
            trial_level_features.append(total_num_words)

            for cluster_by in ["LengthCategory", "POS"]:
                grouped_means = trial.groupby(cluster_by)[
                    ["IA_DWELL_TIME", "IA_FIRST_FIXATION_DURATION"]
                ].transform("mean")
                for et_measure in ["IA_DWELL_TIME", "IA_FIRST_FIXATION_DURATION"]:
                    trial[f"{cluster_by}_normalized_{et_measure}"] = (
                        trial[et_measure] / grouped_means[et_measure]
                    )

            """
            No. values in each groupby type:
            Is_Content_Word 2 (in beyelstm originally 3)
            Reduced_POS 5 (in beyelstm originally 5)
            Entity 20 (in beyelstm originally 11)
            POS 17 (in beyelstm originally 16)
            """
            for groupby_type_, groupby_fields in groupby_mappings:
                grouped_gsf_features = trial.groupby(groupby_type_)[gsf_features].mean()
                melted_gsf_features = ETDataset.add_missing_categories_and_flatten(
                    grouped_gsf_features=grouped_gsf_features,
                    groupby_fields=groupby_fields,
                    groupby_type_=groupby_type_,
                )
                trial_level_features.extend(melted_gsf_features.to_list())
                # assert no nans
                nan_columns = grouped_gsf_features.columns[
                    grouped_gsf_features.isnull().any()
                ]
                if not nan_columns.empty:
                    grouped_gsf_features_with_nans = grouped_gsf_features[nan_columns]
                    warnings.warn(
                        f"NaNs in {groupby_type_} features: {grouped_gsf_features_with_nans}"
                    )
                assert not np.isnan(
                    trial_level_features
                ).any(), "There are NaNs in the data."
        elif mode == ItemLevelFeaturesModes.LENNA:
            # mean saccade duration -> mean "NEXT_SAC_DURATION"
            to_compute_features = [
                "NEXT_SAC_DURATION",
                "NEXT_SAC_AVG_VELOCITY",
                "NEXT_SAC_AMPLITUDE",
            ]
            for feature_to_compute in to_compute_features:
                trial_level_features.append(trial[feature_to_compute].mean())
                trial_level_features.append(trial[feature_to_compute].max())
        elif mode == ItemLevelFeaturesModes.DIANE:
            # mean fixation duration -> mean "CURRENT_FIX_DURATION"
            trial_level_features.append(trial["CURRENT_FIX_DURATION"].mean())

            # mean forward saccade lenght:
            # * "normalized_ID+1" = "normalized_ID" of the next fixation (row)
            trial["normalized_ID+1"] = trial["normalized_ID"].shift(-1)
            # * mean "NEXT_SAC_AMPLITUDE" where "normalized_ID+1" > "normalized_ID"
            forward_saccade_length = trial[
                trial["normalized_ID+1"] > trial["normalized_ID"]
            ]["NEXT_SAC_AMPLITUDE"].mean()
            trial_level_features.append(forward_saccade_length)

            # regression rate - backward saccade rate
            # * using "normalized_ID+1" = "normalized_ID" of the next fixation (row)
            # * regressopn rate - % of rows where "normalized_ID+1" < "normalized_ID"
            regression_rate = (
                trial["normalized_ID+1"] < trial["normalized_ID"]
            ).sum() / len(trial)
            trial_level_features.append(regression_rate)
        elif mode == ItemLevelFeaturesModes.TOTAL_READING_TIME:
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return pd.Series(trial_level_features)

    @staticmethod
    def compute_ia_trial_level_features(
        trial: pd.DataFrame,
        mode: ItemLevelFeaturesModes = ItemLevelFeaturesModes.DAVID,
    ) -> pd.Series:
        trial_level_features = []
        if mode == ItemLevelFeaturesModes.DAVID:
            pass
        elif mode == ItemLevelFeaturesModes.LENNA:
            # skip rate -> mean "IA_DWELL_TIME" == 0
            trial_level_features.append(trial["IA_DWELL_TIME"].eq(0).mean())

            # number of fixations -> sum "IA_FIXATION_COUNT"
            trial_level_features.append(trial["IA_FIXATION_COUNT"].sum())

            # reading speed = mean dwell time
            trial_level_features.append(trial["IA_DWELL_TIME"].mean())
        elif mode == ItemLevelFeaturesModes.DIANE:
            # reading time = words per minute -> num_of_words / "PARAGRAPH_RT"
            # num_of_words is the number of rows in the trial
            num_of_words = len(trial)
            reading_time = num_of_words / trial["PARAGRAPH_RT"].values[0]
            trial_level_features.append(reading_time)

            # first pass skip rate -> mean "IA_SKIP"
            trial_level_features.append(trial["IA_SKIP"].mean())

            # mean first fixation duration -> mean "IA_FIRST_FIXATION_DURATION"
            trial_level_features.append(trial["IA_FIRST_FIXATION_DURATION"].mean())

            # mean gaze duration -> mean "IA_FIRST_RUN_DWELL_TIME"
            trial_level_features.append(trial["IA_FIRST_RUN_DWELL_TIME"].mean())

            # mean word reading time -> mean "IA_DWELL_TIME"
            trial_level_features.append(trial["IA_DWELL_TIME"].mean())

            # mean go past time -> mean "IA_FIRST_RUN_DWELL_TIME"
            trial_level_features.append(trial["IA_FIRST_RUN_DWELL_TIME"].mean())
        elif mode == ItemLevelFeaturesModes.TOTAL_READING_TIME:
            # total reading time -> sum "IA_DWELL_TIME"
            trial_level_features.append(trial["IA_DWELL_TIME"].sum())
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return pd.Series(trial_level_features)

    def compute_trial_level_features_parallel(
        self,
        raw_fixation_data: pd.DataFrame,
        raw_ia_data: pd.DataFrame,
        trial_groupby_columns: list[str],
    ) -> pd.DataFrame:
        # pass the function with the mode as a partial function
        compute_ia_trial_level_features_func = partial(
            self.compute_ia_trial_level_features, mode=self.item_level_features_mode
        )
        compute_fixation_trial_level_features_func = partial(
            self.compute_fixation_trial_level_features,
            mode=self.item_level_features_mode,
        )

        try:  # 22 vs 137 seconds
            print("Using swifter for parallel processing of trial level features.")
            fixation_data_trial_level_features = raw_fixation_data.swifter.groupby(
                trial_groupby_columns
            ).apply(compute_fixation_trial_level_features_func)
            ia_data_trial_level_features = raw_ia_data.swifter.groupby(
                trial_groupby_columns
            ).apply(compute_ia_trial_level_features_func)

        except Exception as e:
            print(  # TODO not supported anymore, use another library
                "Swift not found. Install `mamba install -c conda-forge swifter` and `pip install ray` to use it."
            )
            print(e)
            fixation_data_trial_level_features = raw_fixation_data.groupby(
                trial_groupby_columns
            ).apply(compute_fixation_trial_level_features_func)
            ia_data_trial_level_features = raw_ia_data.groupby(
                trial_groupby_columns
            ).apply(compute_ia_trial_level_features_func)

        # merge the two trial level features
        trial_level_features = pd.concat(
            [fixation_data_trial_level_features, ia_data_trial_level_features],
            axis=1,
        )
        # reset column names to 0, 1, 2, ...
        trial_level_features.columns = list(range(len(trial_level_features.columns)))

        return trial_level_features  # type: ignore


def add_missing_features(
    ia_data: pd.DataFrame, trial_groupby_columns: list
) -> pd.DataFrame:
    """
    Add and transform features in the given DataFrame.

    This function adds and transforms several features in the DataFrame. It also creates
    new features based on existing ones.

    Args:
        ia_data (pd.DataFrame): The input DataFrame. It should have the following columns:
            - Reduced_POS
            - Is_Content_Word
            - NEXT_FIX_INTEREST_AREA_INDEX
            - CURRENT_FIX_INTEREST_AREA_INDEX
            - IA_REGRESSION_IN_COUNT
            - IA_REGRESSION_OUT_FULL_COUNT
            - IA_FIXATION_COUNT
        trial_groupby_columns (list): A list of column names to group by when calculating sums.

    Returns:
        pd.DataFrame: The DataFrame with added and transformed features. The function creates the following new features:
            - Reduced_POS: Transformed from categorical to numerical using a mapping dictionary.
            - Is_Content_Word: Converted to integer type.
            - is_reg: Indicates whether the next fixation interest area index is less than the current one.
            - is_progressive: Indicates whether the next fixation interest area index is greater than the current one.
            - is_reg_sum: The sum of is_reg for each group defined by trial_groupby_columns.
            - is_progressive_sum: The sum of is_progressive for each group defined by trial_groupby_columns.
            - IA_REGRESSION_IN_COUNT_sum: The sum of IA_REGRESSION_IN_COUNT for each group defined by trial_groupby_columns.
            - normalized_outgoing_regression_count: The ratio of IA_REGRESSION_OUT_FULL_COUNT to is_reg_sum.
            - normalized_outgoing_progressive_count: The ratio of the difference between IA_FIXATION_COUNT and IA_REGRESSION_OUT_FULL_COUNT to is_progressive_sum.
            - normalized_incoming_regression_count: The ratio of IA_REGRESSION_IN_COUNT to IA_REGRESSION_IN_COUNT_sum.
            # These are used for Syntactic Clusters with Universal Dependencies PoS and Information Clusters [Berzak et al. 2017]
            - LengthCategory: The length category of the word based on the Length column.
            - LengthCategory_normalized_IA_DWELL_TIME: IA_DWELL_TIME normalized by the mean IA_DWELL_TIME of the LengthCategory group.
            - POS_normalized_IA_DWELL_TIME: IA_DWELL_TIME normalized by the mean IA_DWELL_TIME of the POS group.
            - LengthCategory_normalized_IA_FIRST_FIXATION_DURATION: IA_FIRST_FIXATION_DURATION normalized by the mean IA_FIRST_FIXATION_DURATION of the LengthCategory group.
            - POS_normalized_IA_FIRST_FIXATION_DURATION: IA_FIRST_FIXATION_DURATION normalized by the mean IA_FIRST_FIXATION_DURATION of the POS group.
    """
    # Map Reduced_POS values to numbers
    value_to_number = {"FUNC": 0, "NOUN": 1, "VERB": 2, "ADJ": 3, "UNKNOWN": 4}
    ia_data["Reduced_POS"] = ia_data["Reduced_POS"].map(value_to_number)

    # Convert Is_Content_Word to integer
    ia_data["Is_Content_Word"] = ia_data["Is_Content_Word"].astype(int)

    # Add is_reg and is_progressive features
    ia_data["is_reg"] = (
        ia_data["NEXT_FIX_INTEREST_AREA_INDEX"]
        < ia_data["CURRENT_FIX_INTEREST_AREA_INDEX"]
    )
    ia_data["is_progressive"] = (
        ia_data["NEXT_FIX_INTEREST_AREA_INDEX"]
        > ia_data["CURRENT_FIX_INTEREST_AREA_INDEX"]
    )

    # Calculate sums for is_reg, is_progressive, and IA_REGRESSION_IN_COUNT
    grouped_sums = ia_data.groupby(trial_groupby_columns)[
        ["is_reg", "is_progressive", "IA_REGRESSION_IN_COUNT"]
    ].transform("sum")

    # Add sum features
    ia_data["is_reg_sum"] = grouped_sums["is_reg"]
    ia_data["is_progressive_sum"] = grouped_sums["is_progressive"]
    ia_data["IA_REGRESSION_IN_COUNT_sum"] = grouped_sums["IA_REGRESSION_IN_COUNT"]

    # Add normalized count features
    ia_data["normalized_outgoing_regression_count"] = (
        ia_data["IA_REGRESSION_OUT_FULL_COUNT"] / ia_data["is_reg_sum"]
    )
    ia_data["normalized_outgoing_progressive_count"] = (
        ia_data["IA_FIXATION_COUNT"] - ia_data["IA_REGRESSION_OUT_FULL_COUNT"]
    ) / ia_data["is_progressive_sum"]  # approximation
    ia_data["normalized_incoming_regression_count"] = (
        ia_data["IA_REGRESSION_IN_COUNT"] / ia_data["IA_REGRESSION_IN_COUNT_sum"]
    )

    # Define the boundaries of the bins by word length
    bins = [0, 2, 5, 11, np.inf]  # 0-1, 2-4, 5-10, 11+
    # Define the labels for the bins
    labels = [0, 1, 2, 3]
    ia_data["LengthCategory"] = pd.cut(
        ia_data["Length"], bins=bins, labels=labels, right=False
    )

    # fillna with 0 for the relevant columns
    ia_data.fillna(
        {
            "normalized_outgoing_regression_count": 0,
            "normalized_outgoing_progressive_count": 0,
            "normalized_incoming_regression_count": 0,
            "LengthCategory_normalized_IA_DWELL_TIME": 0,
            "POS_normalized_IA_DWELL_TIME": 0,
            "LengthCategory_normalized_IA_FIRST_FIXATION_DURATION": 0,
            "POS_normalized_IA_FIRST_FIXATION_DURATION": 0,
        },
        inplace=True,
    )

    return ia_data
