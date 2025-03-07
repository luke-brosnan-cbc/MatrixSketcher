# sketchlib/cur_decomposition.py


import numpy as np
from numpy.random import default_rng
from scipy.linalg import pinv, svd
from scipy.sparse import isspmatrix, csr_matrix
from scipy.sparse.linalg import svds
from ._utils import _validate_rank


def cur_decomposition(X, rank, random_state=None, sampling="uniform", regularization=0.0):
    """
    CUR decomposition with optional leverage score sampling and regularization.
    ...
    """
    rng = default_rng(random_state)
    n, p = X.shape

    rank = _validate_rank(rank, min(n, p), "cur_decomposition")
    if sampling not in {"uniform", "leverage"}:
        raise ValueError("sampling must be 'uniform' or 'leverage'")

    # Column selection
    if sampling == "leverage":
        if isspmatrix(X):
            _, s, Vt = svds(X, k=min(rank, min(X.shape) - 1))
        else:
            _, s, Vt = svd(X, full_matrices=False)
        Vt = Vt[np.argsort(s)[::-1], :]
        col_probs = np.sum(Vt[:rank].T**2, axis=1)
        col_probs /= np.sum(col_probs)
        col_indices = rng.choice(p, rank, replace=False, p=col_probs)
    else:
        col_indices = rng.choice(p, rank, replace=False)

    # Row selection
    if sampling == "leverage":
        if isspmatrix(X):
            U, s, _ = svds(X, k=min(rank, min(X.shape) - 1))
        else:
            U, s, _ = svd(X, full_matrices=False)
        U = U[:, np.argsort(s)[::-1]]
        row_probs = np.sum(U**2, axis=1)
        row_probs /= np.sum(row_probs)
        row_indices = rng.choice(n, rank, replace=False, p=row_probs)
    else:
        row_indices = rng.choice(n, rank, replace=False)

    # Build C, R, W
    if isspmatrix(X):
        C = X[:, col_indices].tocsc()
        R = X[row_indices, :].tocsr()
        W = X[row_indices, :][:, col_indices].toarray()
    else:
        C = X[:, col_indices]
        R = X[row_indices, :]
        W = X[np.ix_(row_indices, col_indices)]

    if regularization > 0:
        W += regularization * np.eye(W.shape[0])
    W_pinv = pinv(W)

    return C.dot(W_pinv).dot(R)
