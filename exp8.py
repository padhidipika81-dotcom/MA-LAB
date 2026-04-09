from sklearn_extra.cluster import KMedoids
import numpy as np

X = np.array([[2,3],[3,4],[6,7],[7,8]])

model = KMedoids(n_clusters=2, random_state=0)
model.fit(X)

print("Clusters:", model.labels_)
print("Medoids:", model.cluster_centers_)
