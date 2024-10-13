"""Data module for creating the data."""

import pickle
from dataclasses import asdict
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import process_onestop_sr_report.preprocessing as prp
from pytorch_metric_learning import samplers
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs.constants import NUM_FOLDS, Fields
from src.configs.main_config import Args
from src.ETDataset import ETDataset
from src.FoldSplitter import FoldSplitter
from src.TextDataSet import TextDataSet


class ETDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning data module for the eye tracking data.
    """

    TEST_SPLIT_MODE = "test"
    VAL_SPLIT_MODE = "val"

    def __init__(self, cfg: Args):
        super().__init__()
        self.cfg = cfg
        text_dataset_name = (
            f"v2_TextDataSet_maxlen={self.cfg.model.max_seq_len}_"
            f"prependeyes={self.cfg.model.model_params.prepend_eye_data}_"
            f"addanswers={self.cfg.model.prediction_config.add_answers}_"
            f"predmode={self.cfg.model.prediction_config.prediction_mode}_"
            f"concat_or_duplicate={self.cfg.model.model_params.concat_or_duplicate}_"
            f"preorder={self.cfg.model.preorder}"
        )
        self.text_dataset_path = self.cfg.data_path.text_data_path.with_name(
            text_dataset_name
        ).with_suffix(".pkl")

        self.save_hyperparameters(asdict(self.cfg))

    def prepare_data(self) -> None:
        if self.cfg.data.overwrite_data or not self.text_dataset_path.exists():
            print(
                f"Creating textDataSet and saving to pkl file: {self.text_dataset_path}"
            )
            # create and save to pkl
            text_data = TextDataSet(cfg=self.cfg)
            with open(self.text_dataset_path, "wb") as f:
                pickle.dump(text_data, f)
        else:
            print(
                f"TextDataSet already exists at: {self.text_dataset_path} and overwrite is False"
            )

    def setup(self, stage: str | None = None) -> None:
        ia_data, fixation_data = self.load_data()
        grouped_ia_data = ia_data.groupby(self.cfg.data.groupby_columns)
        ia_group_keys = pd.DataFrame(
            data=list(grouped_ia_data.groups), columns=self.cfg.data.groupby_columns
        )

        grouped_fixation_data = None
        if fixation_data is not None:
            grouped_fixation_data = fixation_data.groupby(self.cfg.data.groupby_columns)
            fixation_group_keys = pd.DataFrame(
                data=list(grouped_fixation_data.groups),
                columns=self.cfg.data.groupby_columns,
            )
            # check that the fixation data and IA data have the same keys
            assert fixation_group_keys[self.cfg.data.groupby_columns].equals(
                ia_group_keys[self.cfg.data.groupby_columns]
            ), "Fixation and IA data have different keys"

        splitter = FoldSplitter(
            item_columns=self.cfg.data.item_defining_columns,
            subject_column=self.cfg.data.subject_column,
            groupby_columns=self.cfg.data.groupby_columns,
            num_folds=NUM_FOLDS,
            use_double_test_size=self.cfg.data.use_double_test_size,
            target_column=self.cfg.model.prediction_config.target_column,
            stratify=self.cfg.data.stratify,
        )
        if self.cfg.data.resplit_items_subjects:
            splitter.create_folds(ia_group_keys)
        # If not, then the folds should already exist and
        # TODO add verification that subjects/items in folds are the same as in the data

        train_keys, val_keys_list, test_keys_list = splitter._train_val_test_splits(
            group_keys=ia_group_keys,
            fold_index=self.cfg.data.fold_index,
        )

        text_data = self.load_text_dataset()

        ia_scaler = self.cfg.data.normalization_type.value()
        fixation_scaler = self.cfg.data.normalization_type.value()
        trial_features_scaler = self.cfg.data.normalization_type.value()
        self.train_dataset = self._create_etdataset(
            keys=train_keys,
            grouped_ia_data=grouped_ia_data,
            grouped_fixation_data=grouped_fixation_data,
            raw_ia_data=ia_data,
            raw_fixation_data=fixation_data,
            text_data=text_data,
            ia_scaler=ia_scaler,
            fixation_scaler=fixation_scaler,
            trial_features_scaler=trial_features_scaler,
        )

        check_is_fitted(ia_scaler)
        if self.cfg.model.use_fixation_report:
            check_is_fitted(fixation_scaler)

        if stage == "fit":
            self.val_datasets = [
                self._create_etdataset(
                    keys=val_group_keys,
                    grouped_ia_data=grouped_ia_data,
                    grouped_fixation_data=grouped_fixation_data,
                    raw_ia_data=ia_data,
                    raw_fixation_data=fixation_data,
                    text_data=text_data,
                    ia_scaler=ia_scaler,
                    fixation_scaler=fixation_scaler,
                    trial_features_scaler=trial_features_scaler,
                )
                for val_group_keys in val_keys_list
            ]

        if stage == "test":
            self.test_datasets = [
                self._create_etdataset(
                    keys=test_group_keys,
                    grouped_ia_data=grouped_ia_data,
                    grouped_fixation_data=grouped_fixation_data,
                    raw_ia_data=ia_data,
                    raw_fixation_data=fixation_data,
                    text_data=text_data,
                    ia_scaler=ia_scaler,
                    fixation_scaler=fixation_scaler,
                    trial_features_scaler=trial_features_scaler,
                )
                for test_group_keys in test_keys_list
            ]

        if stage == "predict":
            self.val_datasets = [
                self._create_etdataset(
                    keys=val_group_keys,
                    grouped_ia_data=grouped_ia_data,
                    grouped_fixation_data=grouped_fixation_data,
                    raw_ia_data=ia_data,
                    raw_fixation_data=fixation_data,
                    text_data=text_data,
                    ia_scaler=ia_scaler,
                    fixation_scaler=fixation_scaler,
                    trial_features_scaler=trial_features_scaler,
                )
                for val_group_keys in val_keys_list
            ]

            self.test_datasets = [
                self._create_etdataset(
                    keys=test_group_keys,
                    grouped_ia_data=grouped_ia_data,
                    grouped_fixation_data=grouped_fixation_data,
                    raw_ia_data=ia_data,
                    raw_fixation_data=fixation_data,
                    text_data=text_data,
                    ia_scaler=ia_scaler,
                    fixation_scaler=fixation_scaler,
                    trial_features_scaler=trial_features_scaler,
                )
                for test_group_keys in test_keys_list
            ]

    def create_dataloader(
        self, dataset, shuffle, do_fancy_sampling: bool = False
    ) -> DataLoader:
        if do_fancy_sampling:
            sampler = samplers.MPerClassSampler(
                labels=self.train_dataset.abcd_labels,  # TODO generalize name (see in ETDataset)
                m=1,
                length_before_new_iter=self.cfg.trainer.samples_per_epoch,  # number of trials in epoch
            )
            shuffle = None

        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.cfg.model.batch_size,
            num_workers=self.cfg.trainer.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        return self.create_dataloader(
            self.train_dataset,
            shuffle=True,
            do_fancy_sampling=self.cfg.trainer.do_fancy_sampling,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            self.create_dataloader(dataset, shuffle=False)
            for dataset in self.val_datasets
        ]

        # predict_dataloaders.append(
        #     DataLoader(
        #         ConcatDataset(self.test_datasets),
        #         batch_size=self.cfg.model.batch_size,
        #         num_workers=self.cfg.trainer.num_workers,
        #         shuffle=False,
        #     )
        # )

    def test_dataloader(self) -> list[DataLoader]:
        return [
            self.create_dataloader(dataset, shuffle=False)
            for dataset in self.test_datasets
        ]

    def predict_dataloader(self) -> list[DataLoader]:
        return [
            self.create_dataloader(dataset, shuffle=False)
            for dataset in self.val_datasets + self.test_datasets
        ]

    @staticmethod
    def _get_data(
        raw_data: pd.DataFrame,
        groups: dict[tuple, pd.Index],
        group_keys: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extracts data from a Pandas DataFrame based on group keys.

        Args:
            raw_data (pd.DataFrame): The original data from which to extract data.
            groups (dict): A dictionary that maps group names to indices.
            group_keys (list): A list of group names to extract data for.

        Returns:
            pd.DataFrame: A new DataFrame containing the extracted data.

        """
        data_indices_union = pd.Index([], dtype="int64")
        for key_ in tqdm(
            group_keys.itertuples(name=None, index=False),
            total=len(group_keys),
            desc="Extracting data",
        ):
            indices = groups[key_]
            data_indices_union = data_indices_union.union(indices)
        return raw_data.loc[data_indices_union].copy()

    def _create_etdataset(
        self,
        grouped_ia_data: pd.core.groupby.DataFrameGroupBy,  # type: ignore
        grouped_fixation_data: pd.core.groupby.DataFrameGroupBy | None,  # type: ignore
        keys: pd.DataFrame,
        raw_ia_data: pd.DataFrame,
        raw_fixation_data: pd.DataFrame | None,
        text_data: TextDataSet,
        ia_scaler: MinMaxScaler | RobustScaler | StandardScaler,
        fixation_scaler: MinMaxScaler | RobustScaler | StandardScaler | None,
        trial_features_scaler: MinMaxScaler | RobustScaler | StandardScaler | None,
    ) -> ETDataset:
        """
        Returns an instance of ETDataset for the given indices, using provided data.

        Args:
            grouped_data (pd.core.groupby.DataFrameGroupBy): A DataFrameGroupBy object containing
            the raw data grouped by subject and unique item.
            keys (pd.DataFrame): A DataFrame containing the group keys to extract data for.
            raw_data (pd.DataFrame): A DataFrame containing the raw data.
            name (str, optional): The name of the dataset. Defaults to None.

        Returns:
            ETDataset: An instance of ETDataset containing the specified data.
        """
        df_ia = ETDataModule._get_data(
            raw_data=raw_ia_data, groups=grouped_ia_data.groups, group_keys=keys
        )

        df_fixation = None
        if grouped_fixation_data is not None and raw_fixation_data is not None:
            df_fixation = ETDataModule._get_data(
                raw_data=raw_fixation_data,
                groups=grouped_fixation_data.groups,
                group_keys=keys,
            )
        return ETDataset(
            ia_data=df_ia,
            fixation_data=df_fixation,
            text_data=text_data,
            ia_scaler=ia_scaler,
            fixation_scaler=fixation_scaler,
            trial_features_scaler=trial_features_scaler,
            regime_name=keys.attrs["name"],
            cfg=self.cfg,
            set_name=keys.attrs["set_name"],
        )

    def unpack_unique_item_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.cfg.data.unique_item_columns] = df[
            self.cfg.data.unique_item_column
        ].str.split("_", expand=True)
        return df

    def load_text_dataset(self) -> TextDataSet:
        textdataset_path = self.text_dataset_path
        print(f"Loading textDataSet from pkl file: {textdataset_path}")
        with open(textdataset_path, "rb") as f:
            text_data = pickle.load(f)
        return text_data

    def load_report_dataset(
        self, dataset_path: Path, query: str | None
    ) -> pd.DataFrame:
        data = prp.load_data(dataset_path, has_preview_to_numeric=True)
        data = self.unpack_unique_item_column(df=data)

        print(f"Number of entries: {len(data)}")
        if query is not None:
            data = data.query(query)
            print(f"Query: {query}")
            print(f"Number of entries after query: {len(data)}")
        else:
            # print a warning no query
            print(f"***** No query for {dataset_path.name} *****")

        return data

    def add_ia_report_features_to_fixation_data(
        self, ia_data: pd.DataFrame, fixation_data: pd.DataFrame
    ) -> pd.DataFrame:
        # TODO complicated code, I think can be simplified by using on_left and on_right and
        # keeping only the columns in ia_data that we want to add to fixation_data + groupby columns
        # Can use errors='ignore' to ignore columns that don't exist in ia_data
        ia_data_renamed = ia_data.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )
        ia_data_renamed = ia_data_renamed[
            self.cfg.data.groupby_columns
            + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
            + self.cfg.model.ia_features_to_add_to_fixation_data
        ]
        common_columns = list(
            set(fixation_data.columns).intersection(ia_data_renamed.columns)
        )
        common_columns = [
            col
            for col in common_columns
            if col
            not in self.cfg.data.groupby_columns
            + [Fields.FIXATION_REPORT_IA_ID_COL_NAME]
        ]
        ia_data_renamed = ia_data_renamed.drop(columns=common_columns)
        # TODO there seems to be nans in normalized_part_id, we don't use it here so discarding but get back to this.
        if (
            "normalized_part_ID" in fixation_data.columns
            and fixation_data["normalized_part_ID"].isna().any()
        ):
            print("WARNING: normalized_part_ID contains nans; dropping it.")
        fixation_data = fixation_data.drop(
            columns=["normalized_part_ID"], errors="ignore"
        )
        fixation_data = fixation_data.merge(
            ia_data_renamed,
            on=self.cfg.data.groupby_columns + [Fields.FIXATION_REPORT_IA_ID_COL_NAME],
            how="left",
            validate="many_to_one",
        )

        return fixation_data

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Throughout the whole script if cfg.model.use_fixation_report
        is false then fixation_data is None"""
        # TODO consider loading only the data that is needed (cols/rows) and to use parquet / dask.
        # TODO pyarrow already considerably improves compared to without.
        fixation_data = None
        ia_data = self.load_report_dataset(
            self.cfg.data_path.et_data_path, self.cfg.data.ia_query
        )
        if self.cfg.model.use_fixation_report:
            fixation_data = self.load_report_dataset(
                self.cfg.data_path.fixations_enriched_path, self.cfg.data.fixation_query
            )
            fixation_data = self.add_ia_report_features_to_fixation_data(
                ia_data, fixation_data
            )
        return ia_data, fixation_data


class ETDataModuleFast(ETDataModule):
    """
    A subclass of ETDataModule that includes checks to prevent redundant data preparation and setup.
    This class is based on the solution provided in https://github.com/Lightning-AI/pytorch-lightning/issues/16005


    Attributes:
        prepare_data_done (bool): A flag indicating whether the prepare_data method has been called.
        setup_stages_done (set): A set storing the stages for which the setup method has been called.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the ETDataModuleFast instance.

        Args:
            *args: Variable length argument list to be passed to the ETDataModule constructor.
            **kwargs: Arbitrary keyword arguments to be passed to the ETDataModule constructor.
        """
        super().__init__(*args, **kwargs)
        self.prepare_data_done = False
        self.setup_stages_done = set()

    def prepare_data(self) -> None:
        """
        Prepare data for the module. If this method has been called before, it does nothing.
        """
        if not self.prepare_data_done:
            super().prepare_data()
            self.prepare_data_done = True

    def setup(self, stage: str) -> None:
        """
        Set up the module for a specific stage. If this method has been called before for the same stage, it does nothing.

        Args:
            stage (str): The stage for which to set up the module.
        """
        if stage not in self.setup_stages_done:
            super().setup(stage)
            self.setup_stages_done.add(stage)
