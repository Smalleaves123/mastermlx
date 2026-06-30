"""mastermlx: a NumPy-first machine learning playground."""

from .version import __version__
from .config import get_backend, set_backend

from .anomaly import *  # noqa: F401,F403
from .bandits import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403
from .decomposition import *  # noqa: F401,F403
from .ensemble import *  # noqa: F401,F403
from .linear_models import *  # noqa: F401,F403
from .manifold import *  # noqa: F401,F403
from .neighbors import *  # noqa: F401,F403
from .neural_net import *  # noqa: F401,F403
from .nlp import *  # noqa: F401,F403
from .probabilistic import *  # noqa: F401,F403
from .rl import *  # noqa: F401,F403
from .selection import *  # noqa: F401,F403
from .semi_supervised import *  # noqa: F401,F403
from .signal import *  # noqa: F401,F403
from .robotics import *  # noqa: F401,F403
from .svm import *  # noqa: F401,F403
from .trees import *  # noqa: F401,F403
from .variational import *  # noqa: F401,F403
from .viz import *  # noqa: F401,F403
from .preprocessing import *  # noqa: F401,F403
from .math_tools import *  # noqa: F401,F403

from .clustering import *  # noqa: F401,F403  # overrides BayesGMM/VGMM with clustering aliases
from .utils import create_rng, log_sum_exp, set_seed

from . import anomaly as _anomaly
from . import bandits as _bandits
from . import clustering as _clustering
from . import data as _data
from . import decomposition as _decomposition
from . import ensemble as _ensemble
from . import linear_models as _linear_models
from . import math_tools as _math_tools
from . import manifold as _manifold
from . import neighbors as _neighbors
from . import neural_net as _neural_net
from . import nlp as _nlp
from . import probabilistic as _probabilistic
from . import rl as _rl
from . import robotics as _robotics
from . import selection as _selection
from . import semi_supervised as _semi_supervised
from . import preprocessing as _preprocessing
from . import signal as _signal
from . import svm as _svm
from . import trees as _trees
from . import variational as _variational
from . import viz as _viz


def _extend_unique(names, items):
    for name in items:
        if name not in names:
            names.append(name)


__all__ = ["__version__", "get_backend", "set_backend", "create_rng", "set_seed", "log_sum_exp"]
_extend_unique(__all__, _anomaly.__all__)
_extend_unique(__all__, _bandits.__all__)
_extend_unique(__all__, _data.__all__)
_extend_unique(__all__, _decomposition.__all__)
_extend_unique(__all__, _ensemble.__all__)
_extend_unique(__all__, _linear_models.__all__)
_extend_unique(__all__, _math_tools.__all__)
_extend_unique(__all__, _manifold.__all__)
_extend_unique(__all__, _neighbors.__all__)
_extend_unique(__all__, _neural_net.__all__)
_extend_unique(__all__, _nlp.__all__)
_extend_unique(__all__, _probabilistic.__all__)
_extend_unique(__all__, _rl.__all__)
_extend_unique(__all__, _preprocessing.__all__)
_extend_unique(__all__, _selection.__all__)
_extend_unique(__all__, _semi_supervised.__all__)
_extend_unique(__all__, _signal.__all__)
_extend_unique(__all__, _robotics.__all__)
_extend_unique(__all__, _svm.__all__)
_extend_unique(__all__, _trees.__all__)
_extend_unique(__all__, [name for name in _variational.__all__ if name not in {"BayesGMM", "VGMM"}])
_extend_unique(__all__, _viz.__all__)
_extend_unique(__all__, _clustering.__all__)
