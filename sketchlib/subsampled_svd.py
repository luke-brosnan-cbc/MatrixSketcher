# sketchlib/subsampled_svd.py


import numpy as np
from numpy.random import default_rng
from scipy.linalg import svd
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import svds

from ._utils import _validate_rank


def subsampled_svd(X, rank, random_state=None, use_partial_svd=True):
    """
    Subsample rows and compute low-rank approximation via SVD.
    ...
    """
    rng = default_rng(random_state)
    n, p = X.shape

    rank = _validate_rank(rank, min(n, p), "subsampled_svd")

    row_indices = rng.choice(n, rank, replace=False)
    X_sampled = X[row_indices, :] if not isspmatrix(X) else X[row_indices, :].copy()

    if use_partial_svd and rank < min(X_sampled.shape):
        U, S, Vt = svds(X_sampled, k=rank)
        idx = np.argsort(S)[::-1]
        S, U, Vt = S[idx], U[:, idx], Vt[idx, :]
    else:
        U, S, Vt = svd(X_sampled, full_matrices=False)

    actual_rank = min(rank, len(S))
    S_inv = np.diag(1.0 / S[:actual_rank])
    Vt_top = Vt[:actual_rank, :]

    return (X.dot(Vt_top.T)).dot(S_inv).dot(Vt_top)
