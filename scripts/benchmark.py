"""
Benchmark mastermlx vs sklearn on common ML tasks.
Requires: pip install mastermlx[compare]
"""
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

METRICS = {}


def bench(name, fn, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    METRICS[name] = avg
    return avg


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def row(name, ours, ref=None):
    tag = f"{name:35s} {ours:8.4f}s"
    if ref is not None:
        ratio = ours / ref if ref > 0 else 0
        tag += f"  vs sklearn {ref:8.4f}s  ({ratio:.1f}x)"
    print(tag)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
header("Classification — 5000 samples, 20 features")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(5000, 20, n_informative=10, random_state=42)
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)


def bench_clf(name, ours_cls, sklearn_cls=None):
    ours = bench(f"{name} (ours)", lambda: ours_cls.fit(Xt, yt).score(Xv, yv))
    ref = None
    if sklearn_cls is not None:
        sk = bench(f"{name} (sklearn)", lambda: sklearn_cls.fit(Xt, yt).score(Xv, yv))
        ref = sk
    row(name, ours, ref)


from mastermlx import LogisticRegression, SGDClassifier, RandomForestClassifier, SVC
from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.linear_model import SGDClassifier as SkSGD
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.svm import SVC as SkSVC

bench_clf("LogisticRegression", LogisticRegression(lr=0.1, n_iter=100), SkLR(max_iter=100))
bench_clf("SGDClassifier", SGDClassifier(loss="hinge", max_iter=50), SkSGD(loss="hinge", max_iter=50))
bench_clf("RandomForest (30 trees)", RandomForestClassifier(30, max_depth=10, random_state=0),
          SkRF(30, max_depth=10, random_state=0))
bench_clf("SVC (RBF)", SVC(kernel="rbf", max_iter=500), SkSVC(kernel="rbf", max_iter=500))


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
header("Clustering — 3000 samples, 10 features, 5 clusters")

from sklearn.datasets import make_blobs
Xb, _ = make_blobs(3000, 10, centers=5, random_state=42)

from mastermlx import KMeans, DBSCAN, GMM
from sklearn.cluster import KMeans as SkKM, DBSCAN as SkDB
from sklearn.mixture import GaussianMixture as SkGMM


def bench_cluster(name, ours_fn, sklearn_fn=None):
    ours_t = bench(f"{name} (ours)", ours_fn)
    ref = None
    if sklearn_fn is not None:
        ref = bench(f"{name} (sklearn)", sklearn_fn)
    row(name, ours_t, ref)


bench_cluster("KMeans (k=5)", lambda: KMeans(5, random_state=0).fit(Xb),
              lambda: SkKM(5, random_state=0, n_init=1).fit(Xb))
bench_cluster("DBSCAN", lambda: DBSCAN(eps=1.5, min_samples=5).fit(Xb),
              lambda: SkDB(eps=1.5, min_samples=5).fit(Xb))
bench_cluster("GMM (5 components)", lambda: GMM(5, random_state=0).fit(Xb),
              lambda: SkGMM(5, random_state=0).fit(Xb))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
header("Regression — 3000 samples, 15 features")

from sklearn.datasets import make_regression
Xr, yr = make_regression(3000, 15, n_informative=10, noise=1.0, random_state=42)

from mastermlx import LinearRegression, RidgeRegression
from sklearn.linear_model import LinearRegression as SkLinR, Ridge as SkRidge

bench_clf("LinearRegression", LinearRegression(), SkLinR())
bench_clf("Ridge (alpha=1)", RidgeRegression(alpha=1.0), SkRidge(alpha=1.0))


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
header("Decomposition — 2000x50 -> 5 components")

from mastermlx import PCA, NMF
from sklearn.decomposition import PCA as SkPCA, NMF as SkNMF

Xpca = np.random.randn(2000, 50)
bench_cluster("PCA (5 comp)", lambda: PCA(5).fit_transform(Xpca),
              lambda: SkPCA(5).fit_transform(Xpca))
Xpca_abs = np.abs(Xpca)
bench_cluster("NMF (5 comp)", lambda: NMF(5, max_iter=200, random_state=0).fit_transform(Xpca_abs),
              lambda: SkNMF(5, max_iter=200, random_state=0).fit_transform(Xpca_abs))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header("Summary")
print(f"{'Task':35s} {'mastermlx':>10s} {'sklearn':>10s} {'Ratio':>8s}")
print("-" * 65)
total_ours = total_sk = 0.0
count = 0
for name, t in sorted(METRICS.items()):
    if "(ours)" in name:
        base = name.replace(" (ours)", "")
        sk_name = base + " (sklearn)"
        ours_t = t
        sk_t = METRICS.get(sk_name, None)
        if sk_t is not None:
            total_ours += ours_t
            total_sk += sk_t
            count += 1
            print(f"{base:35s} {ours_t:10.4f}s {sk_t:10.4f}s {sk_t/ours_t:7.1f}x")

if count > 0:
    print("-" * 65)
    print(f"{'TOTAL':35s} {total_ours:10.4f}s {total_sk:10.4f}s {total_sk/total_ours:7.1f}x")
