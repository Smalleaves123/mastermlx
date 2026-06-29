"""Ensemble learning methods."""

from .bagging import BaggingClassifier, BaggingRegressor
from .hist_gb import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from .extra_trees import ExtraTreesClassifier, ExtraTreesRegressor
from .stacking import StackingClassifier, StackingRegressor
from .calibrate import CalibratedClassifierCV
from .multioutput import MultiOutputClassifier, MultiOutputRegressor
from .voting import VotingClassifier, VotingRegressor

__all__ = [
    "BaggingClassifier",
    "BaggingRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "MultiOutputClassifier",
    "MultiOutputRegressor",
    "CalibratedClassifierCV",
    "VotingClassifier",
    "VotingRegressor",
]
