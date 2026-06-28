"""Ensemble learning methods."""

from .bagging import BaggingClassifier, BaggingRegressor
from .extra_trees import ExtraTreesClassifier, ExtraTreesRegressor
from .stacking import StackingClassifier, StackingRegressor
from .voting import VotingClassifier, VotingRegressor

__all__ = [
    "BaggingClassifier",
    "BaggingRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "VotingClassifier",
    "VotingRegressor",
]
