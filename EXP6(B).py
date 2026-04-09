import numpy as np
from sklearn.cluster import KMeans

# Data points
X = np.array([2, 4, 10, 12, 3, 20, 30, 11, 25]).reshape(-1, 1)

# Apply K-Means
kmeans = KMeans(n_clusters=2, init=np.array([[2],[20]]), n_init=1)
kmeans.fit(X)

# Results
print("Clusters:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)
