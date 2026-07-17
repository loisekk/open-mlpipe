from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    ID_LIKE = "id_like"
    CONSTANT = "constant"
    BOOLEAN = "boolean"


class MissingStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    KNN = "knn"
    MICE = "mice"
    CONSTANT = "constant"
    AUTO = "auto"


class OutlierMethod(str, Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    NONE = "none"
    AUTO = "auto"


class ScalerType(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    POWER = "power"
    NONE = "none"
    AUTO = "auto"


class EncoderType(str, Enum):
    ONEHOT = "onehot"
    TARGET = "target"
    ORDINAL = "ordinal"
    FREQUENCY = "frequency"
    LABEL = "label"
    AUTO = "auto"
