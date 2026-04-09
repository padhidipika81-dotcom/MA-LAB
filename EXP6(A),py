import numpy as np
from sklearn.cluster import KMeans

# Data points
X = np.array([
    [2, 3],   # A
    [3, 4],   # B
    [6, 6],   # C
    [7, 7]    # D
])

# Apply K-Means
kmeans = KMeans(n_clusters=2, init=np.array([[2,3],[6,6]]), n_init=1)
kmeans.fit(X)

# Results
print("Clusters:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)
