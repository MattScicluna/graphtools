from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent


class KNNBackend(ABC):
    """Abstract base for k-NN back-ends."""

    def __init__(self, data, n_neighbors, metric="euclidean", **kw):
        self.data = data
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._fit_backend(**kw)

    # ------------------------------------------------------------------ #
    # each concrete subclass implements these four:
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _fit_backend(self, **kw):             pass

    @abstractmethod
    def set_params(self, **kw):               pass

    @abstractmethod
    def kneighbors(self, X, n_neighbors):     pass

    @abstractmethod
    def kneighbors_graph(self, X, n_neighbors, mode):  pass

    # radius search is optional; default = brute.
    def radius_neighbors(self, X, radius):
        # generic slow fallback (sufficient for the rare path)
        dists   = np.linalg.norm(self.data[None, :, :] - X[:, None, :], axis=-1)
        mask    = dists < radius
        rows, cols = np.nonzero(mask)
        return [dists[i, mask[i]].astype(np.float32) for i in range(len(X))], \
               [cols[rows == i] for i in range(len(X))]

    # brute-force fallback used when search_knn grows huge
    def brute_force_knntree(self, n_neighbors, n_jobs=1):
        return SklearnKNNBackend(self.data, n_neighbors, metric=self.metric,
                                 algorithm="brute", n_jobs=n_jobs)


class SklearnKNNBackend(KNNBackend):
    def _fit_backend(self, algorithm="auto", n_jobs=1):
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=algorithm,
            metric=self.metric,
            n_jobs=n_jobs,
        ).fit(self.data)

    def set_params(self, **kw):                       
        self.model.set_params(**kw)
    def kneighbors(self, X, n_neighbors):     
        print('running sklearn kneighbors')    
        return self.model.kneighbors(X, n_neighbors)
    def kneighbors_graph(self, X, n_neighbors, mode): 
        print('running sklearn kneighbors_graph')
        return self.model.kneighbors_graph(X, n_neighbors, mode)
    def radius_neighbors(self, X, radius):
        print('running sklearn radius neighbors')            
        return self.model.radius_neighbors(X, radius)


class PynndescentKNNBackend(KNNBackend):
    def _fit_backend(self, n_jobs=1, **kw):
        n_trees = min(64, 5 + int(round((self.data.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(self.data.shape[0]))))

        self.index = NNDescent(
            self.data,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=n_jobs,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            low_memory=False,
            verbose=False,
            compressed=False,
            **kw,
        )

    def set_params(self, **kw):
        # NNDescent has no set_params; rebuild if n_jobs changes
        if "n_jobs" in kw:
            self._fit_backend(**kw)

    def kneighbors(self, X, n_neighbors):
        print('Running pynndescent kneighbors')
        inds, dists = self.index.query(X, k=n_neighbors)
        return dists.astype(np.float32), inds

    def kneighbors_graph(self, X, n_neighbors, mode="connectivity"):
        print('Running pynndescent kneighbors_graph')
        dists, inds = self.kneighbors(X, n_neighbors)
        rows = np.repeat(np.arange(X.shape[0]), n_neighbors)
        cols = inds.ravel()
        if mode == "connectivity":
            data = np.ones_like(cols, dtype=np.float32)
        else:  # 'distance'
            data = dists.ravel()
        return csr_matrix((data, (rows, cols)), shape=(X.shape[0], self.data.shape[0]))

    def radius_neighbors(self, X, radius):
        # approximate: query many neighbours then mask
        print('Running pynndescent radius neighbors')
        k = min(self.n_neighbors * 5, self.data.shape[0])
        inds, dists = self.index.query(X, k=k)
        mask = dists < radius
        rows = np.array([d[mask[i]] for i, d in enumerate(dists)])
        cols = np.array([ind[mask[i]] for i, ind in enumerate(inds)])
        return rows, cols
