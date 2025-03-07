# sketchlib/sparse_embedding.py


import numpy as np
from numpy.random import default_rng
from scipy.sparse import isspmatrix, random as sparse_random


def sparse_embedding(X, sketch_size, density=0.1, random_state=None):
    """
    Sparse random projection with controlled non-zero density.
    ...
    """
    rng = default_rng(random_state)
    n, p = X.shape

    if not (0 < density <= 1):
        raise ValueError("density must be in (0, 1]")

    S = sparse_random(
        sketch_size, p,
        density=density,
        format="csr",
        random_state=rng.integers(1e9),
        data_rvs=lambda nn: rng.choice([-1, 1], nn)
    )
    return X.dot(S.T) if isspmatrix(X) else X @ S.T
