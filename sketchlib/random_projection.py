# sketchlib/random_projection.py


import numpy as np
from numpy.random import default_rng
from scipy.sparse import isspmatrix

def random_projection(X, target_dim, random_state=None):
    """
    Apply Johnson-Lindenstrauss random projection.
    ...
    """
    rng = default_rng(random_state)
    n, p = X.shape
    R = rng.normal(loc=0.0, scale=1.0, size=(p, target_dim)) / np.sqrt(target_dim)
    return X.dot(R) if isspmatrix(X) else X @ R
