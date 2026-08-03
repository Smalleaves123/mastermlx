"""Semi-supervised learning methods."""

from .label import LabelPropagation, LabelSpreading
from .self_training import SelfTrainingClassifier

__all__ = ["LabelPropagation", "LabelSpreading", "SelfTrainingClassifier"]
