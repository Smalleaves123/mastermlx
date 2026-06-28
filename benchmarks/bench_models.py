"""
Benchmark mastermlx vs sklearn on common ML tasks.
"""
import time, warnings
import numpy as np
warnings.filterwarnings("ignore")

results = {}


def bench(fn, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def compare(name, ours_fn, sklearn_fn=None):
    t_ours = bench(ours_fn)
    line = f"  {name:35s} {t_ours:8.4f}s"
    if sklearn_fn is not None:
        t_sk = bench(sklearn_fn)
        line += f"  vs sklearn {t_sk:8.4f}s  ({t_ours/t_sk:.1f}x)"
        results[name] = (t_ours, t_sk)
    print(line)


# ---------------------------------------------------------------------------
section("Classification — 5000x20")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as sk_split
from mastermlx.data import train_test_split as mlx_split

X, y = make_classification(5000, 20, n_informative=10, random_state=42)
Xt, Xv, yt, yv = sk_split(X, y, test_size=0.2, random_state=42)

from mastermlx import LogisticRegression, SGDClassifier, RandomForestClassifier, SVC
from sklearn.linear_model import LogisticRegression as SkLR, SGDClassifier as SkSGD
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.svm import SVC as SkSVC

compare("LogisticRegression", lambda: LogisticRegression(lr=0.1, n_iter=100).fit(Xt, yt),
        lambda: SkLR(max_iter=100).fit(Xt, yt))
compare("SGDClassifier (hinge)", lambda: SGDClassifier(loss="hinge", max_iter=50).fit(Xt, yt),
        lambda: SkSGD(loss="hinge", max_iter=50).fit(Xt, yt))
compare("RandomForest (30 trees, d=10)", lambda: RandomForestClassifier(30, max_depth=10, random_state=0).fit(Xt, yt),
        lambda: SkRF(30, max_depth=10, random_state=0).fit(Xt, yt))
compare("SVC (RBF)", lambda: SVC(kernel="rbf", max_iter=500).fit(Xt, yt),
        lambda: SkSVC(kernel="rbf", max_iter=500).fit(Xt, yt))


# ---------------------------------------------------------------------------
section("Clustering — 3000x10, 5 clusters")

from sklearn.datasets import make_blobs
Xb, _ = make_blobs(3000, 10, centers=5, random_state=42)

from mastermlx import KMeans, DBSCAN, GMM
from sklearn.cluster import KMeans as SkKM, DBSCAN as SkDB
from sklearn.mixture import GaussianMixture as SkGMM

compare("KMeans (k=5)", lambda: KMeans(5, random_state=0).fit(Xb),
        lambda: SkKM(5, random_state=0, n_init=1).fit(Xb))
compare("DBSCAN", lambda: DBSCAN(eps=1.5, min_samples=5).fit(Xb),
        lambda: SkDB(eps=1.5, min_samples=5).fit(Xb))
compare("GMM (5 comp)", lambda: GMM(5, random_state=0).fit(Xb),
        lambda: SkGMM(5, random_state=0).fit(Xb))


# ---------------------------------------------------------------------------
section("Regression — 3000x15")

from sklearn.datasets import make_regression
Xr, yr = make_regression(3000, 15, n_informative=10, noise=1.0, random_state=42)

from mastermlx import LinearRegression, RidgeRegression
from sklearn.linear_model import LinearRegression as SkLinR, Ridge as SkRidge

compare("LinearRegression", lambda: LinearRegression().fit(Xr, yr),
        lambda: SkLinR().fit(Xr, yr))
compare("Ridge (alpha=1)", lambda: RidgeRegression(alpha=1.0).fit(Xr, yr),
        lambda: SkRidge(alpha=1.0).fit(Xr, yr))


# ---------------------------------------------------------------------------
section("Decomposition — 2000x50 -> 5")

from mastermlx import PCA, NMF
from sklearn.decomposition import PCA as SkPCA, NMF as SkNMF

Xpca = np.random.randn(2000, 50)
Xabs = np.abs(Xpca)

compare("PCA (5 comp)", lambda: PCA(5).fit_transform(Xpca),
        lambda: SkPCA(5).fit_transform(Xpca))
compare("NMF (5 comp)", lambda: NMF(5, max_iter=200, random_state=0).fit_transform(Xabs),
        lambda: SkNMF(5, max_iter=200, random_state=0).fit_transform(Xabs))


# ---------------------------------------------------------------------------
section("Summary")
total_o, total_s = 0.0, 0.0
print(f"  {'Task':35s} {'mastermlx':>10s} {'sklearn':>10s} {'Ratio':>8s}")
print(f"  {'-'*63}")
for name, (o, s) in results.items():
    total_o += o; total_s += s
    print(f"  {name:35s} {o:10.4f}s {s:10.4f}s {o/s:7.1f}x")
print(f"  {'-'*63}")
print(f"  {'TOTAL':35s} {total_o:10.4f}s {total_s:10.4f}s {total_o/total_s:7.1f}x")
