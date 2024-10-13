"""
This module contains configuration classes for various prediction modes.

The module defines a base configuration class `PredCfg` and several derived classes
for specific prediction modes such as question prediction, chosen answer prediction,
correctness prediction, condition prediction, and reread prediction.

Each configuration class is defined using the `@dataclass` decorator and specifies
the relevant attributes and their default values for the corresponding prediction mode.

The configuration classes use constants from the `src.configs.constants` module, namely
`PredMode` and `TargetColumn`, to define the prediction mode and target column respectively.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING

from src.configs.constants import PredMode, TargetColumn


@dataclass
class PredCfg:
    """
    Base configuration class for prediction modes.

    Attributes:
        prediction_mode (PredMode): The prediction mode. Must be specified by derived classes.
        class_names (list[str]): List of class names for the prediction. Default is an empty list.
        target_column (TargetColumn): Target column for the prediction. Must be specified by derived classes.
        use_eyes_only (bool): If True, text data is discarded from ETDataset. Default is False.
        add_answers (bool): Whether to add answers to the text data. Default is True.
    """

    prediction_mode: PredMode = MISSING
    class_names: list[str] = field(default_factory=list)
    target_column: TargetColumn = MISSING
    use_eyes_only: bool = False
    add_answers: bool = True


@dataclass
class QPredCfg(PredCfg):
    """
    Configuration for question prediction mode.

    Inherits from PredCfg.

    Attributes:
        prediction_mode (PredMode): The prediction mode. Default is PredMode.QUESTION_LABEL.
        class_names (list[str]): List of class names for the prediction. Default is ["LonelyQuestion", "CoupledQuestion1", "CoupledQuestion2"].
        target_column (TargetColumn): Target column for the prediction. Default is TargetColumn.Q_IND.
        add_answers (bool): Whether to add answers to the text data. Default is False.

    Inherited attributes from PredCfg:
        use_eyes_only (bool): If True, text data is discarded from ETDataset. Default is False.
    """

    prediction_mode: PredMode = PredMode.QUESTION_LABEL
    class_names: list[str] = field(
        default_factory=lambda: [
            "LonelyQuestion",
            "CoupledQuestion1",
            "CoupledQuestion2",
        ]
    )
    target_column: TargetColumn = TargetColumn.QUESTION_LABEL
    add_answers: bool = False


@dataclass
class QConditionPredCfg(PredCfg):
    """
    Configuration for question condition prediction mode.
    That is, predicting which question, including an option
    for the Gathering condition.

    Attributes are inherited from PredictionModeConfig.
    """

    prediction_mode: PredMode = PredMode.QUESTION_n_CONDITION
    class_names: list[str] = field(
        default_factory=lambda: [
            "LonelyQuestion",
            "CoupledQuestion1",
            "CoupledQuestion2",
            "Gathering",
        ]
    )
    target_column: TargetColumn = TargetColumn.QUESTION_n_CONDITION_LABEL
    add_answers: bool = False


@dataclass
class ChosenAnswerPredCfg(PredCfg):
    """
    Configuration for chosen answer prediction mode.

    Inherits from PredCfg.

    Attributes:
        prediction_mode (PredMode): The prediction mode. Default is PredMode.CHOSEN_ANSWER.
        class_names (list[str]): List of class names for the prediction. Default is ["A", "B", "C", "D"].
        target_column (TargetColumn): Target column for the prediction. Default is TargetColumn.CHOSEN_ANSWER.

    Inherited attributes from PredCfg:
        use_eyes_only (bool): If True, text data is discarded from ETDataset. Default is False.
        add_answers (bool): Whether to add answers to the text data. Default is True.
    """

    prediction_mode: PredMode = PredMode.CHOSEN_ANSWER
    class_names: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    target_column: TargetColumn = TargetColumn.CHOSEN_ANSWER


@dataclass
class SimplifiedChosenAnswerPredCfg(PredCfg):
    """
    Configuration for simplified chosen answer prediction mode.

    Attributes:
        prediction_mode (PredMode): The prediction mode. Default is PredMode.CHOSEN_ANSWER.
        class_names (list[str]): List of class names for the prediction. Default is ["AB", "CD"].
        target_column (TargetColumn): Target column for the prediction. Default is TargetColumn.CHOSEN_ANSWER.

    Inherited attributes from PredCfg:
        use_eyes_only (bool): If True, text data is discarded from ETDataset. Default is False.
        add_answers (bool): Whether to add answers to the text data. Default is True.
    """

    prediction_mode: PredMode = PredMode.CHOSEN_ANSWER
    class_names: list[str] = field(default_factory=lambda: ["AB", "CD"])
    target_column: TargetColumn = TargetColumn.CHOSEN_ANSWER


@dataclass
class IsCorrectPredCfg(PredCfg):
    """
    Configuration for correctness prediction mode.

    Attributes:
        prediction_mode (PredMode): The prediction mode. Default is PredMode.IS_CORRECT.
        class_names (list[str]): List of class names for the prediction. Default is ["BCD", "A"].
        target_column (TargetColumn): Target column for the prediction. Default is TargetColumn.IS_CORRECT.

    Inherited attributes from PredCfg:
        use_eyes_only (bool): If True, text data is discarded from ETDataset. Default is False.
        add_answers (bool): Whether to add answers to the text data. Default is False.
    """

    prediction_mode: PredMode = PredMode.IS_CORRECT
    class_names: list[str] = field(default_factory=lambda: ["BCD", "A"])
    target_column: TargetColumn = TargetColumn.IS_CORRECT
    add_answers: bool = False


@dataclass
class CorrectAnswerPredCfg(PredCfg):
    """
    Configuration for correct answer prediction mode.

    Attributes:
        prediction_mode (PredMode): The prediction mode. Default is PredMode.CORRECT_ANSWER.
        class_names (list[str]): List of class names for the prediction. Default is ["a", "b", "c", "d"].
        target_column (TargetColumn): Target column for the prediction. Default is TargetColumn.CORRECT_ANSWER.

    Inherited attributes from PredCfg:
        use_eyes_only (bool): If True, text data is discarded from ETDataset. Default is False.
        add_answers (bool): Whether to add answers to the text data. Default is True.
    """

    prediction_mode: PredMode = PredMode.CORRECT_ANSWER
    class_names: list[str] = field(default_factory=lambda: ["a", "b", "c", "d"])
    target_column: TargetColumn = TargetColumn.CORRECT_ANSWER


@dataclass
class ConditionPredCfg(PredCfg):
    """
    Configuration for condition prediction mode.

    Attributes:
        prediction_mode (PredMode): The prediction mode. Default is PredMode.CONDITION.
        class_names (list[str]): List of class names for the prediction. Default is ["Gathering", "Hunting"].
        target_column (TargetColumn): Target column for the prediction. Default is TargetColumn.HAS_PREVIEW.

    Inherited attributes from PredCfg:
        use_eyes_only (bool): If True, text data is discarded from ETDataset. Default is False.
        add_answers (bool): Whether to add answers to the text data. Default is True.
    """

    prediction_mode: PredMode = PredMode.CONDITION
    class_names: list[str] = field(default_factory=lambda: ["Gathering", "Hunting"])
    target_column: TargetColumn = TargetColumn.HAS_PREVIEW
    add_answers: bool = False


@dataclass
class RereadPredCfg(PredCfg):
    """
    Configuration for reread prediction mode.

    Attributes:
        prediction_mode (PredMode): The prediction mode. Default is PredMode.REREAD.
        class_names (list[str]): List of class names for the prediction. Default is ["Not Reread", "Reread"].
        target_column (TargetColumn): Target column for the prediction. Default is TargetColumn.REREAD.

    Inherited attributes from PredCfg:
        use_eyes_only (bool): If True, text data is discarded from ETDataset. Default is False.
        add_answers (bool): Whether to add answers to the text data. Default is True.
    """

    prediction_mode: PredMode = PredMode.REREAD
    class_names: list[str] = field(default_factory=lambda: ["Not Reread", "Reread"])
    target_column: TargetColumn = TargetColumn.REREAD
