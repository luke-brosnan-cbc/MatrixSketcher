# sketchlib/row_sampling.py


import numpy as np
from numpy.random import default_rng
from scipy.sparse import isspmatrix

def row_sampling(X, sample_size, random_state=None, weighted=False):
    """
    Row sampling with optional norm-based weighting.
    ...
    """
    rng = default_rng(random_state)
    n, p = X.shape

    if sample_size > n:
        raise ValueError(f"sample_size {sample_size} exceeds {n} rows")

    if weighted:
        if isspmatrix(X):
            row_norms = np.array(X.power(2).sum(axis=1)).ravel()
        else:
            row_norms = np.sum(X**2, axis=1)
        probs = row_norms / np.sum(row_norms)
        indices = rng.choice(n, sample_size, replace=False, p=probs)
    else:
        indices = rng.choice(n, sample_size, replace=False)

    return X[indices].copy() if isspmatrix(X) else X[indices]
