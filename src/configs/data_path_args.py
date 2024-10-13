"""Data path arguments for the eye tracking data."""

from dataclasses import dataclass
from pathlib import Path


from src.configs.constants import ConfigName
from src.configs.utils import register_config

GROUP = ConfigName.DATA_PATH


@dataclass
class DataPathArgs:
    """
    Data path arguments for the eye tracking data.
    """

    data_path: Path = Path(
        "ln_shared_data/onestop/processed"
    )  # Path to the data directory. Can be used specify a common path for all data files.
    et_data_path: Path = (
        data_path / "ia_data_enriched_360_05052024.csv"
    )  # Full path to the interest area report
    fixations_enriched_path: Path = (
        data_path / "fixation_data_enriched_360_05052024.csv"
    )  # Full path to the fixation report
    text_data_path: Path = (
        data_path / "all_dat_files_merged.tsv"  # TODO Can be updated (needed?)
    )  # Full path to the experiment dat files.


@register_config(group=GROUP)
@dataclass
class feb11(DataPathArgs):
    def __post_init__(self) -> None:
        self.et_data_path = self.data_path / "ia_data_enriched_360_110224.csv"
        self.fixations_enriched_path = (
            self.data_path / "fixation_data_enriched_360_110224.csv"
        )


@register_config(group=GROUP)
@dataclass
class march31(DataPathArgs):
    def __post_init__(self) -> None:
        self.et_data_path = self.data_path / "ia_data_enriched_360_31032024.csv"
        self.fixations_enriched_path = (
            self.data_path / "fixation_data_enriched_360_31032024.csv"
        )


@register_config(group=GROUP)
@dataclass
class april14(DataPathArgs):
    """
    This includes also the 'q_condition' column in the data.
    """

    def __post_init__(self) -> None:
        self.et_data_path = self.data_path / "ia_data_enriched_360_14042024.csv"
        self.fixations_enriched_path = (
            self.data_path / "fixation_data_enriched_360_14042024.csv"
        )


@register_config(group=GROUP)
@dataclass
class may05(DataPathArgs):
    """
    Rebased Question Prediction and Question_n_Condition Prediction
    Added 'question_label' and 'question_n_condition_label' columns in the data.
    :: question_label: 0 for the lonely question, 1 for the first question in the couple, 2 for the second question in the couple.
    :: question_n_condition_label: question_label if Hunting, 3 if Gathering.
    Removed 'q_condition' column from the data.
    """

    def __post_init__(self) -> None:
        self.et_data_path = self.data_path / "ia_data_enriched_360_05052024.csv"
        self.fixations_enriched_path = (
            self.data_path / "fixation_data_enriched_360_05052024.csv"
        )
