import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
np.random.randn(6)
centers = np.random.randn(6).rehape(3,2)
centers = np.random.randn(6).reshape(3,2)
centers
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)
X
labels
labels_true
plt.ion()
plt.scatter(X[labels_true==0, 0], X[labels_true==0, 1], color='red', alpha=0.5)
plt.scatter(X[labels_true==1, 0], X[labels_true==1, 1], color='blue', alpha=0.5)
plt.scatter(X[labels_true==2, 0], X[labels_true==2, 1], color='green', alpha=0.5)
af = AffinityPropagation(preference=-50).fit(X)
af
cluster_centers_indices = af.cluster_centers_indices_
cluster_centers_indices
labels = af.labels_
labels
labels_true == labels
np.sum(labels_true == labels)
np.sum(labels_true == labels) / labels_true.shape[0]
n_clusters_ = len(cluster_centers_indices)
n_clusters_
metrics.homogeneity_score(labels_true, labels)
metrics.completeness_score(labels_true, labels)
metrics.v_measure_score(labels_true, labels)
metrics.adjusted_rand_score(labels_true, labels)
metrics.adjusted_mutual_info_score(labels_true, labels)
metrics.silhouette_score(X, labels, metric='sqeuclidean')
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
%hist -f silhouette_score_blobs.py
