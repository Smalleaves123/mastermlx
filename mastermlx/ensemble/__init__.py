"""Ensemble learning methods."""

from .bagging import BaggingClassifier, BaggingRegressor
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
    "StackingClassifier",
    "StackingRegressor",
    "MultiOutputClassifier",
    "MultiOutputRegressor",
    "CalibratedClassifierCV",
    "VotingClassifier",
    "VotingRegressor",
]
