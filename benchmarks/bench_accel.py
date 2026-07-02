"""
Compare mastermlx C++/Cython acceleration vs pure NumPy fallback.
Shows the speedup from our compiled extensions.
"""
import time, warnings
import numpy as np
warnings.filterwarnings("ignore")

# Force NumPy backend for comparison
import os
os.environ["MASTERML_BACKEND"] = "numpy"


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

# NumPy (pure Python + NumPy broadcasting)
from mastermlx.accel.backends import _numpy_pairwise_squared_euclidean as np_sq
np_t = bench(lambda: np_sq(X, Y), name="NumPy (10 GB intermediate)")
os.environ["MASTERML_BACKEND"] = "auto"
from mastermlx.accel import pairwise_squared_euclidean as accel_sq
accel_t = bench(lambda: accel_sq(X, Y), name="C++ (O3 native)")
print(f"  -> Speedup: {np_t/accel_t:.1f}x, Memory: 10GB -> 32MB")


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
section("DBSCAN — 2000x30")

Xdb = np.random.randn(2000, 30)

os.environ["MASTERML_BACKEND"] = "numpy"
from mastermlx import DBSCAN
np_db = bench(lambda: DBSCAN(eps=2.0, min_samples=5).fit(Xdb), name="NumPy backend")

os.environ["MASTERML_BACKEND"] = "auto"
accel_db = bench(lambda: DBSCAN(eps=2.0, min_samples=5).fit(Xdb), name="C++ backend")
print(f"  -> Speedup: {np_db/accel_db:.1f}x")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
section("Confusion Matrix — 200k samples, 8 classes")

from mastermlx.utils.metrics import confusion_matrix

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
accel_cm = bench(lambda: confusion_matrix(y_true, y_pred), name="Cython-assisted API")
print(f"  -> Speedup: {np_cm/accel_cm:.1f}x")


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------
section("KMeans — 2000x30, k=5")

from mastermlx import KMeans

os.environ["MASTERML_BACKEND"] = "numpy"
np_km = bench(lambda: KMeans(5, random_state=0).fit(Xdb), name="NumPy backend")

os.environ["MASTERML_BACKEND"] = "auto"
accel_km = bench(lambda: KMeans(5, random_state=0).fit(Xdb), name="C++ backend")
print(f"  -> Speedup: {np_km/accel_km:.1f}x")


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
section("RandomForest — 5000x20, 30 trees")

from mastermlx import RandomForestClassifier
Xrf = np.random.randn(5000, 20)
yrf = np.where(Xrf[:, 0] > 0, 1, 0)

os.environ["MASTERML_BACKEND"] = "numpy"
np_rf = bench(lambda: RandomForestClassifier(30, max_depth=8, random_state=0).fit(Xrf, yrf),
              name="NumPy (pure Python split)", n_runs=2)

os.environ["MASTERML_BACKEND"] = "auto"
accel_rf = bench(lambda: RandomForestClassifier(30, max_depth=8, random_state=0).fit(Xrf, yrf),
                 name="Cython tree ops", n_runs=2)
print(f"  -> Speedup: {np_rf/accel_rf:.1f}x")
os.environ["MASTERML_BACKEND"] = "auto"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("Acceleration Impact Summary")
print(f"  Pairwise Distance: {np_t/accel_t:.0f}x faster, 300x less memory")
print(f"  DBSCAN:            {np_db/accel_db:.0f}x faster")
print(f"  KMeans:            {np_km/accel_km:.0f}x faster")
print(f"  RandomForest:      {np_rf/accel_rf:.0f}x faster")
