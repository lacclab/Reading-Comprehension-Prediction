"""Constants used throughout the project."""

from enum import Enum, StrEnum

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

NUM_ANSWERS = 4
NUM_QUESTIONS = 3
# query to filter longest: (subject_id != 'l31_388' | unique_paragraph_id != '3_1_Adv_4')
MAX_SCANPATH_LENGTH = (
    815  # 814 is the longest scanpath in the dataset. 553 is the second longest.
)
# TODO make MAX_SCANPATH_LENGTH dynamic, like max_seq_len
SCANPATH_PADDING_VAL = -1
NUM_FOLDS = 10

#### For BEyeLSTMModel
Is_Content_Word = [0, 1]
Reduced_POS = [0, 1, 2, 3, 4]
entity_list = [
    "",
    "DATE",
    "QUANTITY",
    "ORG",
    "GPE",
    "NORP",
    "PERSON",
    "TIME",
    "CARDINAL",
    "ORDINAL",
    "PRODUCT",
    "FAC",
    "LAW",
    "LOC",
    "EVENT",
    "MONEY",
    "PERCENT",
    "WORK_OF_ART",
    "LANGUAGE",
    "NaN",
]
pos_list = [
    "PUNCT",
    "PROPN",
    "NOUN",
    "PRON",
    "VERB",
    "SCONJ",
    "NUM",
    "DET",
    "CCONJ",
    "ADP",
    "AUX",
    "ADV",
    "ADJ",
    "INTJ",
    "X",
    "PART",
    "SYM",
]
groupby_mappings = [
    ("Is_Content_Word", Is_Content_Word),
    ("Reduced_POS", Reduced_POS),
    ("Entity", entity_list),
    ("POS", pos_list),
]
gsf_features = [
    "gpt2_Surprisal",
    "Length",
    "n_Lefts",
    "n_Rights",
    "Distance2Head",
    "IA_FIRST_FIXATION_DURATION",
    "IA_DWELL_TIME",
    "normalized_incoming_regression_count",
    "CURRENT_FIX_X",
    "CURRENT_FIX_Y",
    "normalized_outgoing_regression_count",
    "normalized_outgoing_progressive_count",
    "LengthCategory_normalized_IA_DWELL_TIME",
    "POS_normalized_IA_DWELL_TIME",
    "LengthCategory_normalized_IA_FIRST_FIXATION_DURATION",
    "POS_normalized_IA_FIRST_FIXATION_DURATION",
]


class DataRepresentation(StrEnum):
    """
    Enum for MAG model modes.

    Attributes
    ----------
    CONCAT : str
        Represents the mode where data is concatenated.
    DUPLICATE : str
        Represents the mode where data is duplicated.
    """

    CONCAT = "concat"
    DUPLICATE = "duplicate"


class NormalizationModes(StrEnum):
    """
    Enum for normalization modes.

    Attributes
    ----------
    ALL : str
        Represents the mode where data is normalized based on all trials.
    TRIAL : str
        Represents the mode where data is normalized based on a trial level.
    NONE : str
        Represents the mode where no data is normalized.
    """

    ALL = "all"
    TRIAL = "trial"
    NONE = "none"


class RunModes(StrEnum):
    """
    Enum for run modes.

    Attributes
    ----------
    DEBUG : str
        Represents the debug mode. This mode is used for debugging the code.
    FAST_DEV_RUN : str
        Represents the fast development run mode. This mode is used for quickly testing the code.
    TRAIN : str
        Represents the train mode. This mode is used for training the model.
    """

    DEBUG = "debug"
    FAST_DEV_RUN = "fast_dev_run"
    TRAIN = "train"


class Accelerators(StrEnum):
    """
    Enum for accelerator types.

    Attributes
    ----------
    AUTO : str
        Represents the automatic selection of accelerator based on availability.
    CPU : str
        Represents the Central Processing Unit as the accelerator.
    GPU : str
        Represents the Graphics Processing Unit as the accelerator.
    """

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class Fields(StrEnum):
    """Enum for field names in the data."""

    BATCH = "batch"
    PARAGRAPH_ID = "paragraph_id"
    ARTICLE_ID = "article_id"
    Q_IND = "q_ind"
    QUESTION = "question"
    ANSWER = "answer"
    LEVEL = "level"
    LIST = "list"
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    CORRECT_ANSWER = "correct_answer"
    PARAGRAPH = "paragraph"
    ANSWERS_ORDER = "answers_order"
    HAS_PREVIEW = "has_preview"
    SUBJECT_ID = "subject_id"
    FINAL_ANSWER = "FINAL_ANSWER"
    REREAD = "reread"
    IA_DATA_IA_ID_COL_NAME = "IA_ID"
    FIXATION_REPORT_IA_ID_COL_NAME = "CURRENT_FIX_INTEREST_AREA_INDEX"
    IS_CORRECT = "is_correct"
    PRACTICE = "practice"
    ABCD_ANSWER = "abcd_answer"
    Q_CONDITION = "q_condition"
    QUESTION_LABEL = "question_prediction_label"
    QUESTION_n_CONDITION_LABEL = "question_n_condition_prediction_label"


class TargetColumn(StrEnum):
    """
    Enum for target column.

    Attributes
    ----------
    CORRECT_ANSWER : str
        Represents the column for the correct answer.
    HAS_PREVIEW : str
        Represents the column for the preview availability.
    REREAD : str
        Represents the column for the reread status.
    FINAL_ANSWER : str
        Represents the column for the final answer.
    Q_IND : str
        Represents the column for the question index.
    IS_CORRECT : str
        Represents the column for the correctness status.
    Q_CONDITION : str
        Represents the column for the question condition
        [question including an option for condition (i.e., Gathring)].
    """

    CORRECT_ANSWER = Fields.CORRECT_ANSWER
    HAS_PREVIEW = Fields.HAS_PREVIEW
    REREAD = Fields.REREAD
    CHOSEN_ANSWER = Fields.FINAL_ANSWER
    Q_IND = Fields.Q_IND
    IS_CORRECT = Fields.IS_CORRECT
    Q_CONDITION = Fields.Q_CONDITION
    QUESTION_LABEL = Fields.QUESTION_LABEL
    QUESTION_n_CONDITION_LABEL = Fields.QUESTION_n_CONDITION_LABEL


class ItemLevelFeaturesModes(StrEnum):
    DAVID = "David(BEyeLSTM)"
    LENNA = "Lenna"
    DIANE = "Diane"
    TOTAL_READING_TIME = "TotalReadingTime"
    NONE = "None"


class BackboneNames(StrEnum):
    """
    Enum for backbone names.

    Attributes
    ----------
    ROBERTA_BASE : str
        Represents the base version of the RoBERTa model.
    ROBERTA_LARGE : str
        Represents the large version of the RoBERTa model.
    ROBERTA_RACE: str
        Represents the large version of a fine-tuned-on-RACE RoBERTA model.
    """

    ROBERTA_BASE = "roberta-base"
    ROBERTA_LARGE = "roberta-large"
    ROBERTA_RACE = "LIAMF-USP/roberta-large-finetuned-race"


class ModelNames(StrEnum):
    """
    Enum for model names.

    Attributes
    ----------
    LIT_MODEL : str
        Represents the name of the LIT model.
    MLP_MODEL : str
        Represents the name of the MLP (Multilayer Perceptron) model.
    MAG_MODEL : str
        Represents the name of the MAG model.
    FSE_MODEL : str
        Represents the name of the FSE model.
    TOTAL_RT_MLP_MODEL : str
        Represents the name of the Total RT MLP model.
    ROBERTEYE_MODEL : str
        Represents the name of the Eye BERT model.
    EYETTENTION : str
        Representes the name for the Eyettention model, based on https://github.com/aeye-lab/Eyettention
    """

    LIT_MODEL = "LIT_MODEL"
    MLP_MODEL = "MLP_MODEL"
    MAG_MODEL = "MAG_MODEL"
    FSE_MODEL = "FSE_MODEL"
    TOTAL_RT_MLP_MODEL = "TOTAL_RT_MLP_MODEL"
    ROBERTEYE_MODEL = "ROBERTEYE_MODEL"
    AHN_CNN_MODEL = "AHN_CNN_MODEL"
    AHN_RNN_MODEL = "AHN_RNN_MODEL"
    BEYELSTM_MODEL = "BEYELSTM_MODEL"
    EYETTENTION_MODEL = "EYETTENTION_MODEL"
    POSTFUSION_MODEL = "POSTFUSION_MODEL"


class MLModelNames(StrEnum):
    """
    Enum for ML model names.
    """

    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
    KNN = "KNN"
    SVM = "SVM"
    RANDOM_FOREST = "RANDOM_FOREST"
    DUMMY_CLASSIFIER = "DUMMY_CLASSIFIER"


class PredMode(StrEnum):
    """
    Enum for prediction mode.

    Attributes
    ----------
    CHOSEN_ANSWER : str
        Represents the mode where the chosen answer is predicted.
    CORRECT_ANSWER : str
        Represents the mode where the correct answer is predicted.
    QUESTION : str
        Represents the mode where the question is predicted.
    CONDITION : str
        Represents the mode where the condition is predicted.
    REREAD : str
        Represents the mode where reread is predicted.
    IS_CORRECT : str
        Represents the mode where the correctness is predicted.
    Q_CONDITION : str
        Represents the mode where the question including an option for condition (i.e., Gathring) is predicted.
    """

    CHOSEN_ANSWER = "CHOSEN_ANSWER"
    CORRECT_ANSWER = "CORRECT_ANSWER"
    QUESTION = "QUESTION"
    CONDITION = "CONDITION"
    REREAD = "REREAD"
    IS_CORRECT = "IS_CORRECT"
    QUESTION_LABEL = "QUESTION_LABEL"
    QUESTION_n_CONDITION = "QUESTION_n_CONDITION"


class Precision(StrEnum):
    """
    Enum for precision types.

    16_MIXED: Corresponds to "16-mixed"
    32_TRUE: Corresponds to "32-true"
    """

    SIXTEEN_MIXED = "16-mixed"
    THIRTY_TWO_TRUE = "32-true"


class MatmulPrecisionLevel(StrEnum):
    """
    Enum for precision levels.

    HIGHEST: Corresponds to "highest"
    HIGH: Corresponds to "high"
    MEDIUM: Corresponds to "medium"
    """

    HIGHEST = "highest"
    HIGH = "high"
    MEDIUM = "medium"


class Scaler(Enum):
    """
    Enum for scaler types. Each scaler type is associated with a corresponding
    scaler class from sklearn.preprocessing.

    MIN_MAX_SCALER: Corresponds to sklearn.preprocessing.MinMaxScaler
    ROBUST_SCALER: Corresponds to sklearn.preprocessing.RobustScaler
    STANDARD_SCALER: Corresponds to sklearn.preprocessing.StandardScaler
    """

    MIN_MAX_SCALER = MinMaxScaler
    ROBUST_SCALER = RobustScaler
    STANDARD_SCALER = StandardScaler


class ConfigName(StrEnum):
    """
    Enum for config names.

    Attributes
    ----------

    DATA : str
        Represents the data config.
    TRAINER : str
        Represents the trainer config.
    MODEL : str
        Represents the model config.
    DATA_PATH : str
        Represents the data path config.
    """

    DATA = "data"
    TRAINER = "trainer"
    MODEL = "model"
    DATA_PATH = "data_path"

class FeatureMode(StrEnum):
    """
    Enum for feature modes.
    
    """

    EYES = "eyes"
    WORDS = "words"
    EYES_WORDS = "eyes_words"