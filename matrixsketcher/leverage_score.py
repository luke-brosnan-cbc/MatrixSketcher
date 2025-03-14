# matrixsketcher/leverage_score.py


import numpy as np
from numpy.random import default_rng
from scipy.linalg import svd
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import svds
from ._utils import _validate_rank


def leverage_score_sampling(X, sample_size, rank=None, random_state=None, 
                            scale_rows=False, sampling="leverage"):
    """
    Row sampling using leverage scores, uniform selection, or weighted norm-based selection.
    
    Parameters:
    - X (array or sparse matrix): Input data matrix (n x p)
    - sample_size (int): Number of rows to sample
    - rank (int, optional): Rank for SVD (only used if sampling="leverage")
    - random_state (int, optional): Seed for reproducibility
    - scale_rows (bool, optional): Whether to scale rows by sqrt(1/prob)
    - sampling (str): "leverage", "uniform", or "weighted"

    Returns:
    - Sampled subset of rows from X (scaled if requested)
    """
    rng = default_rng(random_state)
    n, p = X.shape

    if sample_size > n:
        raise ValueError(f"sample_size {sample_size} exceeds matrix rows {n}")

    if sampling not in {"uniform", "leverage", "weighted"}:
        raise ValueError("sampling must be 'uniform', 'leverage', or 'weighted'")

    if sampling == "leverage":
        use_partial = (rank is not None) and (rank < min(n, p))
        if use_partial:
            rank = _validate_rank(rank, min(n, p), "leverage_score_sampling")
            U, s, _ = svds(X, k=rank) if isspmatrix(X) else svds(X, k=rank)
            U = U[:, np.argsort(s)[::-1]]  # Sort by singular values
        else:
            U = svd(X.toarray() if isspmatrix(X) else X, full_matrices=False)[0]

        leverage_scores = np.sum(U**2, axis=1)
        leverage_probs = leverage_scores / np.sum(leverage_scores)
        selected_rows = rng.choice(n, size=sample_size, replace=False, p=leverage_probs)

    elif sampling == "weighted":
        if isspmatrix(X):
            row_norms = np.array(X.power(2).sum(axis=1)).ravel()
        else:
            row_norms = np.sum(X**2, axis=1)
        weighted_probs = row_norms / np.sum(row_norms)
        selected_rows = rng.choice(n, size=sample_size, replace=False, p=weighted_probs)

    else:  # Uniform sampling
        selected_rows = rng.choice(n, size=sample_size, replace=False)

    if scale_rows:
        scaled_rows = []
        for idx in selected_rows:
            row = X[idx].toarray().ravel() if isspmatrix(X) else X[idx]
            prob = (
                leverage_probs[idx] if sampling == "leverage"
                else weighted_probs[idx] if sampling == "weighted"
                else 1
            )
            scaled_rows.append(row / np.sqrt(prob))
        return np.vstack(scaled_rows)

    return X[selected_rows].copy() if isspmatrix(X) else X[selected_rows]
