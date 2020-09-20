from sklearn.neighbors import NearestNeighbors
samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
neigh = NearestNeighbors(n_neighbors=2)
neigh.fit(samples)
NearestNeighbors(n_neighbors=2)
print(neigh.kneighbors([[0., 0., 0.]]))