from pathlib import Path

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.configs.constants import Fields


class FoldSplitter:
    """
    A class used to split data into folds.

    Attributes
    ----------
    item_columns : list[str]
        The columns that contain the item identifiers.
    subject_column : str
        The column that contains the subject identifiers.
    groupby_columns : list[str]
        The columns used to group the trials.
    num_folds : int
        The number of folds to split the data into.
    stratify : bool
        Whether to stratify the data.

    Methods
    -------
    __init__(self, item_columns: list[str], subject_column: str, groupby_columns: list[str], num_folds: int) -> None:
        Initializes the FoldSplitter with the given item columns, subject column, groupby columns, and number of folds.
    """

    def __init__(
        self,
        item_columns: list[str],
        subject_column: str,
        groupby_columns: list[str],
        num_folds: int,
        target_column: str,
        use_double_test_size: bool = False,
        stratify: bool = False,
    ) -> None:
        """
        Initializes the FoldSplitter with the given item columns, subject column, groupby columns, and number of folds.

        Parameters
        ----------
        item_columns : list[str]
            The columns that contain the item identifiers.
        subject_column : str
            The column that contains the subject identifiers.
        groupby_columns : list[str]
            The columns used to group the trials.
        num_folds : int
            The number of folds to split the data into.
        use_double_test_size : bool
            Whether to use a double test size.
        target_column : str
            The column that contains the target values.
        stratify : bool
            Whether to stratify the data.
        """
        self.item_columns = item_columns
        self.subject_column = subject_column
        self.groupby_columns = groupby_columns
        self.num_folds = num_folds
        self.use_double_test_size = use_double_test_size
        self.target_column = target_column
        self.stratify = stratify

    def get_split_indices(
        self, group_keys: pd.DataFrame, split_indices: pd.Series, is_item: bool
    ) -> pd.Index:
        if is_item:
            column = group_keys[self.item_columns].apply("_".join, axis=1)
        else:
            column = group_keys[self.subject_column]

        return group_keys.loc[column.isin(split_indices)].index

    def load_folds(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        fold_path = Path("data") / "folds"
        subject_folds_path = fold_path / "subjects"
        item_folds_path = fold_path / "items"
        # load all folds
        subject_folds = []
        item_folds = []
        for i in range(self.num_folds):
            subject_fold_path = subject_folds_path / f"fold_{i}.csv"
            item_fold_path = item_folds_path / f"fold_{i}.csv"
            subject_folds.append(
                pd.read_csv(subject_fold_path, header=None).squeeze("columns")
            )
            item_folds.append(
                pd.read_csv(item_fold_path, header=None).squeeze("columns")
            )
        return subject_folds, item_folds

    def get_fold_indices(
        self,
        i: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Given a fold index i within the range [0, 9], return the indices for the test,
        validation, and training sets according to the specified folding strategy.

        Parameters:
        i (int): The fold index (should be between 0 and 9).

        Returns:
        tuple: A tuple containing the test indices (as a list), validation index (as an integer),
        and training indices (as a list).
        """
        if i < 0 or i > self.num_folds - 1:
            raise ValueError("Fold index must be within the range [0, 9].")

        validation_indices = [i]

        # modulo num_folds for the wraparound
        test_indices = [(i + 1) % self.num_folds]
        if self.use_double_test_size:
            test_indices.append((i + 2) % self.num_folds)

        # The rest are training indices
        train_indices = [
            x
            for x in range(self.num_folds)
            if x not in test_indices and x not in validation_indices
        ]
        print(
            f"Test folds: {test_indices}, \
            Validation fold: {validation_indices}, \
            Train folds: {train_indices}"
        )
        return test_indices, validation_indices, train_indices

    def create_folds(self, group_keys: pd.DataFrame) -> None:
        all_folds_subjects = []
        all_folds_items = []
        n_splits = self.num_folds
        for split_ind in range(n_splits):
            fold_subjects = []
            fold_items = []

            for batch_id in group_keys[Fields.BATCH].unique():
                batch_data = group_keys[
                    group_keys[Fields.BATCH] == batch_id
                ].reset_index(drop=True)

                subjects = batch_data[self.subject_column]
                items = batch_data[self.item_columns].apply("_".join, axis=1)
                y = batch_data[self.target_column]

                if self.stratify:
                    splitter = StratifiedGroupKFold(n_splits=n_splits)
                else:
                    splitter = GroupKFold(n_splits=n_splits)

                _, test_subjects_indx = list(
                    splitter.split(subjects, y=y, groups=subjects)
                )[split_ind]
                _, test_items_indx = list(splitter.split(items, y=y, groups=items))[
                    split_ind
                ]

                fold_subjects.extend(subjects[test_subjects_indx])
                fold_items.extend(items[test_items_indx])

            all_folds_subjects.append(fold_subjects)
            all_folds_items.append(fold_items)
        try:
            folds_path = Path(HydraConfig.get().runtime.output_dir) / "folds"
        except Exception:  # no hydra
            print("HydraConfig not found. Using default path.")
            folds_path = Path("data") / "folds"
            print(folds_path)
        subject_folds_path = folds_path / "subjects"
        item_folds_path = folds_path / "items"
        self.item_folds = {}
        self.subject_folds = {}
        for i, (subject_fold, item_fold) in enumerate(
            zip(all_folds_subjects, all_folds_items)
        ):
            item_folds_path.mkdir(parents=True, exist_ok=True)
            subject_folds_path.mkdir(parents=True, exist_ok=True)
            subject_df = pd.DataFrame(sorted(list(set(subject_fold))))
            self.subject_folds[i] = subject_df
            subject_df.to_csv(
                subject_folds_path / f"fold_{i}.csv", header=False, index=False
            )
            item_df = pd.DataFrame(sorted(list(set(item_fold))))
            self.item_folds[i] = item_df
            item_df.to_csv(item_folds_path / f"fold_{i}.csv", header=False, index=False)
            # print(f"Fold {i}:")
            # print(f"Subjects: \n{subject_df}")
            # print(f"Items: \n{item_df}")

    @staticmethod
    def get_combined_indices(fold_dict: dict, folds_indices: list[int]) -> pd.Series:
        """
        Concatenates the fold indices from the fold dictionary based on the given fold indices.

        Args:
            fold_dict (dict): A dictionary containing fold indices as values.
            folds_indices (list): A list of fold indices to be concatenated.

        Returns:
            pd.Series: A series containing the combined fold indices.
        """
        return pd.concat(
            [fold_dict[i] for i in folds_indices], ignore_index=True
        ).squeeze("columns")

    def _train_val_test_splits(
        self,
        group_keys: pd.DataFrame,
        fold_index: int,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame]]:
        """Splits the data into train and test sets.

        Args:
            group_keys (pd.DataFrame):
            mode (str):

        Raises:
            ValueError:

        Returns:
            Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        # create train, val, test sets
        test_indices, val_indices, train_indices = self.get_fold_indices(fold_index)
        # subject_folds, item_folds = self.load_folds()
        subject_folds = self.subject_folds
        item_folds = self.item_folds

        subject_train_indices = self.get_combined_indices(subject_folds, train_indices)
        subject_val_indices = self.get_combined_indices(subject_folds, val_indices)
        subject_test_indices = self.get_combined_indices(subject_folds, test_indices)
        item_train_indices = self.get_combined_indices(item_folds, train_indices)
        item_val_indices = self.get_combined_indices(item_folds, val_indices)
        item_test_indices = self.get_combined_indices(item_folds, test_indices)

        train_subjects_indx = self.get_split_indices(
            group_keys, subject_train_indices, is_item=False
        )
        val_subjects_indx = self.get_split_indices(
            group_keys, subject_val_indices, is_item=False
        )
        test_subjects_indx = self.get_split_indices(
            group_keys, subject_test_indices, is_item=False
        )
        train_items_indx = self.get_split_indices(
            group_keys, item_train_indices, is_item=True
        )
        val_items_indx = self.get_split_indices(
            group_keys, item_val_indices, is_item=True
        )
        test_items_indx = self.get_split_indices(
            group_keys, item_test_indices, is_item=True
        )

        train_indices = np.array(train_subjects_indx.intersection(train_items_indx))

        # Test indices
        seen_subject_unseen_item_test_indices = np.array(
            test_items_indx.intersection(train_subjects_indx.union(val_subjects_indx))
        )
        unseen_subject_seen_item_test_indices = np.array(
            train_items_indx.union(val_items_indx).intersection(test_subjects_indx)
        )
        unseen_subject_unseen_item_test_indices = np.array(
            test_items_indx.intersection(test_subjects_indx)
        )

        # Val indices
        unseen_subject_unseen_item_val_indices = np.array(
            val_subjects_indx.intersection(val_items_indx)
        )
        unseen_subject_seen_item_val_indices = np.array(
            val_subjects_indx.intersection(train_items_indx)
        )
        seen_subject_unseen_item_val_indices = np.array(
            train_subjects_indx.intersection(val_items_indx)
        )

        # assert all data subsets sum to all the data
        assert len(group_keys) == len(train_indices) + len(
            seen_subject_unseen_item_test_indices
        ) + len(unseen_subject_seen_item_test_indices) + len(
            unseen_subject_unseen_item_test_indices
        ) + len(unseen_subject_unseen_item_val_indices) + len(
            unseen_subject_seen_item_val_indices
        ) + len(
            seen_subject_unseen_item_val_indices
        ), "Data subsets do not sum to all the data"

        # # create train keys
        self.assert_no_duplicates(train_indices, "train_indices")
        train_keys = group_keys.iloc[train_indices]
        train_keys.attrs["name"] = "train"
        train_keys.attrs["set_name"] = "train"

        # create validation keys
        test_key_types = [
            ("seen_subject_unseen_item", seen_subject_unseen_item_test_indices),
            ("unseen_subject_seen_item", unseen_subject_seen_item_test_indices),
            ("unseen_subject_unseen_item", unseen_subject_unseen_item_test_indices),
        ]
        test_keys_list = []
        for key_name, indices in test_key_types:
            self.assert_no_duplicates(indices, key_name)
            keys = group_keys.iloc[indices]
            keys.attrs["name"] = key_name
            keys.attrs["set_name"] = "test"
            test_keys_list.append(keys)

        val_keys_list = []
        val_key_types = [
            ("seen_subject_unseen_item", seen_subject_unseen_item_val_indices),
            ("unseen_subject_seen_item", unseen_subject_seen_item_val_indices),
            ("unseen_subject_unseen_item", unseen_subject_unseen_item_val_indices),
        ]
        for key_name, indices in val_key_types:
            self.assert_no_duplicates(indices, key_name)
            keys = group_keys.iloc[indices]
            keys.attrs["name"] = key_name
            keys.attrs["set_name"] = "val"
            val_keys_list.append(keys)

        # print size of each group
        self.print_group_info("Train", train_keys)

        for keys in val_keys_list:
            self.print_group_info(f"Val {keys.attrs['name']}", keys)

        for keys in test_keys_list:
            self.print_group_info(f"Test {keys.attrs['name']}", keys)

        all_keys = pd.concat([train_keys] + val_keys_list + test_keys_list).sort_index()
        self.print_group_info("All", all_keys)
        # assert all_keys[self.groupby_columns].equals(
        #    group_keys[self.groupby_columns]
        # ), "Train, val and test keys do not contain all the data"

        return train_keys, val_keys_list, test_keys_list

    @staticmethod
    def print_group_info(name: str, keys: pd.DataFrame) -> None:
        print(
            f"{name}: # Trials: {len(keys)}. "
            f"# Items: {keys['unique_paragraph_id'].nunique()}; "
            f"# Subjects: {keys['subject_id'].nunique()}"
        )

    @staticmethod
    def assert_no_duplicates(indices, indices_name) -> None:
        assert len(indices) == len(set(indices)), indices_name + " contains duplicates"
