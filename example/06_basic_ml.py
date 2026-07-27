"""A minimal non-robotics example: regression, PCA, and clustering."""

import numpy as np

from common import check_release
from mastermlx import KMeans, LinearRegression, PCA


check_release()
X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])

regression = LinearRegression().fit(X, y)
prediction = regression.predict([[4.0, 4.0]])
print("linear_regression_prediction:", prediction)

pca = PCA(n_components=1).fit(X)
reduced = pca.transform(X)
print("pca_components_shape:", pca.components_.shape)
print("pca_transformed_shape:", reduced.shape)

clusters = KMeans(n_clusters=2, random_state=0, n_init=5).fit(X)
labels = clusters.predict(X)
print("kmeans_labels:", labels)

print("prediction_at_4_4:", prediction)
print("pca_shape:", reduced.shape)
print("cluster_labels:", labels)
