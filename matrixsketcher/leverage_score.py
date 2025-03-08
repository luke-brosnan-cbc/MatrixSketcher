# matrixsketcher/leverage_score.py


import numpy as np
from numpy.random import default_rng
from scipy.linalg import svd
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import svds
from ._utils import _validate_rank


def leverage_score_sampling(X, sample_size, rank=None, random_state=None,
                            scale_rows=False):
    """
    Leverage score row sampling.
    ...
    """
    rng = default_rng(random_state)
    n, p = X.shape

    if sample_size > n:
        raise ValueError(f"sample_size {sample_size} exceeds matrix rows {n}")

    use_partial = (rank is not None) and (rank < min(n, p))
    if use_partial:
        rank = _validate_rank(rank, min(n, p), "leverage_score_sampling")
        U, s, _ = svds(X, k=rank) if isspmatrix(X) else svds(X, k=rank)
        U = U[:, np.argsort(s)[::-1]]
    else:
        U = svd(X.toarray() if isspmatrix(X) else X, full_matrices=False)[0]

    leverage_scores = np.sum(U**2, axis=1)
    leverage_probs = leverage_scores / np.sum(leverage_scores)

    selected_rows = rng.choice(n, size=sample_size, replace=False, p=leverage_probs)

    if scale_rows:
        scaled_rows = []
        for idx in selected_rows:
            row = X[idx].toarray().ravel() if isspmatrix(X) else X[idx]
            scaled_rows.append(row / np.sqrt(leverage_probs[idx]))
        return np.vstack(scaled_rows)

    return X[selected_rows].copy() if isspmatrix(X) else X[selected_rows]
