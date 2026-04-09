from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([[2,3],[3,4],[6,7],[7,8]])

model = AgglomerativeClustering(n_clusters=2)
labels = model.fit_predict(X)

print("Clusters:", labels)
