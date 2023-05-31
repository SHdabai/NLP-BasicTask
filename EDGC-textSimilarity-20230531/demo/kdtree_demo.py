# -- encoding:utf-8 --

import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric
from sklearn.neighbors import NearestNeighbors

X = [
    [1, 1], [1, 2], [1, 3],
    [2, 1], [2, 2], [2, 3],
    [3, 1], [3, 2], [3, 3]
]
tree = KDTree(X, leaf_size=2)  # doctest: +SKIP
dist, ind = tree.query([[1.1, 1.9]], k=1)  # doctest: +SKIP
print(ind)  # indices of 3 closest neighbors
print(dist)  # distances to 3 closest neighbors
