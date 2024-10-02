# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array

class CustomKMeans:
    def __init__(self, n_clusters, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        np.random.seed(self.random_state)
        X = check_array(X, accept_sparse='csr')
        X = X.astype(np.float32)  # Convert to float32 to prevent underflow

        
        # Randomly select initial centroids
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Compute distances to centroids and assign clusters
            distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
            self.labels_ = np.argmin(distances, axis=0)

            # Recompute centroids as the mean of assigned points
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence (no change in centroids)
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.cluster_centers_ = centroids
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

class CustomDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _find_neighbors(self, X, point):
        # Compute distances using broadcasting
        distances = np.linalg.norm(X - point, axis=1)
        return np.where(distances < self.eps)[0]

    def fit(self, X: np.ndarray):
        X = check_array(X)
        X = X.astype(np.float32, copy=False)  # Convert to float32 for efficiency
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)  # Initialize labels to -1 (unclassified)
        cluster_id = 0

        for idx in range(n_samples):
            if labels[idx] != -1:  # Skip if already classified
                continue

            neighbors = self._find_neighbors(X, X[idx])
            if len(neighbors) < self.min_samples:
                labels[idx] = -1  # Mark as noise
                continue

            labels[idx] = cluster_id
            seeds = set(neighbors)
            seeds.discard(idx)

            while seeds:
                current_point = seeds.pop()
                if labels[current_point] == -1:  # If noise, add to cluster
                    labels[current_point] = cluster_id
                if labels[current_point] != -1:  # Skip if already classified
                    continue
                labels[current_point] = cluster_id
                current_point_neighbors = self._find_neighbors(X, X[current_point])
                if len(current_point_neighbors) >= self.min_samples:
                    seeds.update(current_point_neighbors)

            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_