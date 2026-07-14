"""
Compare mastermlx C++/Cython acceleration vs pure NumPy fallback.
Shows the speedup from our compiled extensions.
"""
import time, warnings
import numpy as np
warnings.filterwarnings("ignore")

from mastermlx import set_backend


def bench(fn, n_runs=3, name=""):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    if name:
        print(f"  {name:30s} {avg:8.4f}s")
    return avg


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------
section("Pairwise Distance — 3000x50")

X = np.random.randn(3000, 50).astype(np.float64)
Y = np.random.randn(3000, 50).astype(np.float64)

from mastermlx.accel import pairwise_squared_euclidean
set_backend("numpy")
np_t = bench(lambda: pairwise_squared_euclidean(X, Y), name="NumPy backend")
set_backend("auto")
accel_t = bench(lambda: pairwise_squared_euclidean(X, Y), name="Compiled backend")
print(f"  -> Compiled speedup: {np_t/accel_t:.1f}x")


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
section("DBSCAN — 1200x20")

Xdb = np.random.randn(1200, 20)

set_backend("numpy")
from mastermlx import DBSCAN
np_db = bench(lambda: DBSCAN(eps=2.0, min_samples=5).fit(Xdb), name="NumPy backend", n_runs=1)

set_backend("auto")
accel_db = bench(lambda: DBSCAN(eps=2.0, min_samples=5).fit(Xdb), name="Compiled backend", n_runs=1)
print(f"  -> Speedup: {np_db/accel_db:.1f}x")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
section("Confusion Matrix — 200k samples, 8 classes")

from mastermlx.utils.metrics import confusion_matrix, _cy_confusion_matrix_counts

y_true = np.random.randint(0, 8, size=200_000)
y_pred = np.random.randint(0, 8, size=200_000)


def py_confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((labels.shape[0], labels.shape[0]), dtype=float)
    for yt, yp in zip(y_true, y_pred):
        cm[index[yt], index[yp]] += 1.0
    return cm


np_cm = bench(lambda: py_confusion_matrix(y_true, y_pred), name="Pure Python loop")
cm_name = "Cython-assisted API" if _cy_confusion_matrix_counts is not None else "Python fallback API"
accel_cm = bench(lambda: confusion_matrix(y_true, y_pred), name=cm_name)
print(f"  -> Speedup: {np_cm/accel_cm:.1f}x")


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------
section("KMeans — 1200x20, k=5")

from mastermlx import KMeans

set_backend("numpy")
np_km = bench(lambda: KMeans(5, random_state=0).fit(Xdb), name="NumPy backend")

set_backend("auto")
accel_km = bench(lambda: KMeans(5, random_state=0).fit(Xdb), name="C++ backend")
print(f"  -> Speedup: {np_km/accel_km:.1f}x")


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
section("RandomForest — 1500x12, 12 trees")

from mastermlx import RandomForestClassifier
Xrf = np.random.randn(1500, 12)
yrf = np.where(Xrf[:, 0] > 0, 1, 0)

set_backend("numpy")
np_rf = bench(lambda: RandomForestClassifier(12, max_depth=6, random_state=0).fit(Xrf, yrf),
              name="NumPy (pure Python split)", n_runs=1)

set_backend("auto")
accel_rf = bench(lambda: RandomForestClassifier(12, max_depth=6, random_state=0).fit(Xrf, yrf),
                 name="Cython tree ops", n_runs=1)
print(f"  -> Speedup: {np_rf/accel_rf:.1f}x")
set_backend("auto")


# ---------------------------------------------------------------------------
# Histogram Gradient Boosting
# ---------------------------------------------------------------------------
section("HistGradientBoosting — 1500x20, 30 trees")

from mastermlx import HistGradientBoostingClassifier

rng = np.random.default_rng(0)
Xhg = rng.normal(size=(1500, 20))
yhg = (Xhg[:, 0] - Xhg[:, 1] > 0).astype(int)

set_backend("numpy")
np_hist = bench(
    lambda: HistGradientBoostingClassifier(
        n_estimators=30, max_depth=5, min_samples_leaf=10, random_state=0
    ).fit(Xhg, yhg),
    name="NumPy histogram tree",
    n_runs=1,
)
set_backend("auto")
accel_hist = bench(
    lambda: HistGradientBoostingClassifier(
        n_estimators=30, max_depth=5, min_samples_leaf=10, random_state=0
    ).fit(Xhg, yhg),
    name="C++ histogram tree",
    n_runs=1,
)
print(f"  -> Speedup: {np_hist/accel_hist:.1f}x")
set_backend("auto")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("Acceleration Impact Summary")
print(f"  Pairwise Distance: {np_t/accel_t:.1f}x faster with compiled backend")
print(f"  DBSCAN:            {np_db/accel_db:.1f}x faster")
print(f"  KMeans:            {np_km/accel_km:.1f}x faster")
print(f"  RandomForest:      {np_rf/accel_rf:.1f}x faster")
print(f"  HistGradientBoost: {np_hist/accel_hist:.1f}x faster")
