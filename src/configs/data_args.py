"""Data arguments for the eye tracking data."""

from dataclasses import dataclass, field

from src.configs.constants import ConfigName, Fields, NormalizationModes, Scaler
from src.configs.utils import register_config

GROUP = ConfigName.DATA


@register_config(group=GROUP)
@dataclass
class DataArgs:
    """
    A dataclass for storing configuration parameters for handling eye tracking data.

    Attributes:
        fold_index (int): Defines the test fold. +1 is also test. +2 is validation, rest (out of 10) are train.
        overwrite_data (bool): If True, overwrites the relevant TextDataSet and ETDataset.features even if they exist.
        resplit_items_subjects (bool): If True, resplits items and subjects. Careful if changed to False.
        subject_column (str): Column that defines the subject.
        unique_item_column (str): Column that defines an item. Equivalent to unique_item_columns concatenated and separated by "_".
        normalization_mode (NormalizationModes): Mode for normalization.
        normalization_type (Scaler): Type of scaler for normalization.
        unique_item_columns (list[str]): Columns that make up a unique item.
        ia_query (str | None): Interest area query for filtering rows.
        fixation_query (str | None): Fixation query for filtering rows.
        item_defining_columns (list[str]): Defines item for train-test split grouping (w/out level and paragraph).
        item_subject_column (str): Column that defines the item subject.
        reread_column (str): Column that indicates if the item was reread.
        question_ind_column (str): Column that indicates the question index.
        answer_order_column (str): Column that indicates the order of the answers.
        has_preview_column (str): Column that indicates if the item has a preview.
        groupby_columns (list[str]): Columns used for grouping data. Defined in __post_init__.

    Methods:
        __post_init__: Initializes the groupby_columns attribute based on the values of other attributes.
    """

    # Defines the test fold. +1 is also test. +2 is validation, rest (out of 10) are train.
    fold_index: int = 0

    # If True, overwrites the relevant TextDataSet even if it exists.
    overwrite_data: bool = False

    # Careful if change to False
    resplit_items_subjects: bool = True

    # column that defines the subject
    subject_column: str = Fields.SUBJECT_ID

    # column that defines an item. Equivalent to unique_item_columns concatenated and separated by "_"
    unique_item_column: str = "unique_paragraph_id"

    normalization_mode: NormalizationModes = NormalizationModes.ALL  #! Before changing, make sure eval has been updated to load actual config values and not default
    normalization_type: Scaler = Scaler.ROBUST_SCALER  #!  Before changing, make sure eval has been updated to load actual config values and not default

    # columns that make up a unique item
    unique_item_columns: list[str] = field(
        default_factory=lambda: [
            Fields.BATCH,
            Fields.ARTICLE_ID,
            Fields.LEVEL,
            Fields.PARAGRAPH_ID,
        ]
    )

    # interest area query for filtering rows
    ia_query: str | None = "practice==0"

    # fixation query for filtering rows
    fixation_query: str | None = None

    # defines item for train-test split grouping (w/out level and paragraph)
    item_defining_columns: list[str] = field(
        default_factory=lambda: [
            Fields.BATCH,
            Fields.ARTICLE_ID,
        ]
    )

    item_subject_column: str = Fields.LIST
    reread_column: str = Fields.REREAD
    question_ind_column: str = Fields.Q_IND
    answer_order_column: str = Fields.ANSWERS_ORDER
    has_preview_column: str = Fields.HAS_PREVIEW
    abcd_answer_column: str = Fields.ABCD_ANSWER

    # Whether to double the test size for the test set (2 folds instead of 1)
    use_double_test_size = False

    # Whether to stratify the data based on the target variable
    stratify = True

    # Defined in __post_init__ below
    groupby_columns: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Careful changing this, especially sub-trial (e.g. CS)! Target is added automatically later on.
        self.groupby_columns = (
            [self.answer_order_column]  # 0
            + self.unique_item_columns  # 1-4
            + [
                self.unique_item_column,  # 5
                self.subject_column,  # 6
                self.reread_column,  # 7
                self.item_subject_column,  # 8
                self.question_ind_column,  # 9
                self.has_preview_column,  # 10
                self.abcd_answer_column,  # 11 # TODO assumes abcd answer is last in ETDataset ! not good..!
                # * self.question_condition_column,  # 11
            ]
        )


# ? CURRENT_FIX_INTEREST_AREA_INDEX=-1 is CURRENT_FIX_INTEREST_AREA_LABEL='.' as in they did not fixate on an interest area
@register_config(group=GROUP)
@dataclass
class NoReread(DataArgs):
    """
    A dataclass for storing configuration parameters for handling eye tracking data where reread is False.

    Inherits all attributes from DataArgs and overrides `ia_query` and `fixation_query`.

    Attributes:
        ia_query (str): Interest area query for filtering rows where practice is 0 and reread is 0.
        fixation_query (str): Fixation query for filtering rows where practice is 0, reread is 0, and CURRENT_FIX_INTEREST_AREA_INDEX is greater than or equal to 0.
    """

    ia_query: str = "practice==0 & reread==0"
    fixation_query: str = (
        "practice==0 & reread==0 & CURRENT_FIX_INTEREST_AREA_INDEX >= 0"
    )


@register_config(group=GROUP)
@dataclass
class Hunting(DataArgs):
    """
    A dataclass for storing configuration parameters for handling eye tracking data where has_preview is True.

    Inherits all attributes from DataArgs and overrides `ia_query` and `fixation_query`.

    Attributes:
        ia_query (str): Interest area query for filtering rows where practice is 0, reread is 0, and has_preview is 1.
        fixation_query (str): Fixation query for filtering rows where practice is 0, reread is 0, CURRENT_FIX_INTEREST_AREA_INDEX is greater than or equal to 0, and has_preview is 1.
    """

    ia_query: str = "practice==0 & reread==0 & has_preview==1"
    fixation_query: str = (
        "practice==0 & reread==0 & "
        "CURRENT_FIX_INTEREST_AREA_INDEX >= 0 & has_preview==1"
    )


@register_config(group=GROUP)
@dataclass
class Gathering(DataArgs):
    """
    A dataclass for storing configuration parameters for handling eye tracking data where has_preview is False.

    Inherits all attributes from DataArgs and overrides `ia_query` and `fixation_query`.

    Attributes:
        ia_query (str): Interest area query for filtering rows where practice is 0, reread is 0, and has_preview is 0.
        fixation_query (str): Fixation query for filtering rows where practice is 0, reread is 0, CURRENT_FIX_INTEREST_AREA_INDEX is greater than or equal to 0, and has_preview is 0.
    """

    ia_query: str = "(practice==0 & reread==0 & has_preview==0)"  # & (subject_id != 'l31_388' | unique_paragraph_id != '3_1_Adv_4')"
    fixation_query: str = (
        "(practice==0 & reread==0 & "
        "CURRENT_FIX_INTEREST_AREA_INDEX >= 0 & has_preview==0)"  # & (subject_id != 'l31_388' | unique_paragraph_id != '3_1_Adv_4')"
    )


@register_config(group=GROUP)
@dataclass
class HuntingCSOnly(DataArgs):
    """
    A dataclass for storing configuration parameters for handling eye tracking data where has_preview is True and is_in_aspan is True.

    Inherits all attributes from DataArgs and overrides `ia_query` and `fixation_query`.

    Attributes:
        ia_query (str): Interest area query for filtering rows where practice is 0, reread is 0, has_preview is 1, and is_in_aspan is True.
        fixation_query (str): Fixation query for filtering rows where practice is 0, reread is 0, CURRENT_FIX_INTEREST_AREA_INDEX is greater than or equal to 0, has_preview is 1, and is_in_aspan is True.
    """

    ia_query: str = "practice==0 & reread==0 & has_preview==1 & is_in_aspan==True"
    fixation_query: str = (
        "practice==0 & reread==0 & CURRENT_FIX_INTEREST_AREA_INDEX >= 0 & "
        "has_preview==1  & is_in_aspan==True"
    )
