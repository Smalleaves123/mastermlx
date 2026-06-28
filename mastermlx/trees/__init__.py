"""Tree-based methods."""

from .adaboost import AdaBoostClassifier, AdaBoostClf
from .adaboost_regressor import AdaBoostRegressor
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from .gradient_boosting import GBR, GradientBoostingRegressor
from .gradient_boosting_classifier import GBC, GradientBoostingClassifier
from .random_forest import RFClf, RFReg, RandomForestClassifier, RandomForestRegressor

__all__ = [
    "AdaBoostClassifier",
    "AdaBoostClf",
    "AdaBoostRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GBC",
    "GradientBoostingClassifier",
    "GBR",
    "GradientBoostingRegressor",
    "RFClf",
    "RFReg",
    "RandomForestClassifier",
    "RandomForestRegressor",
]
