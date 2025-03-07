# sketchlib/nystrom.py


import numpy as np
from numpy.random import default_rng
from scipy.linalg import pinv
from ._utils import _validate_rank


def nystrom(
    X_or_K,
    rank,
    kernel=None,
    gamma=None,
    random_state=None,
    return_factorized=False,
    regularization=0.0,
    rbf_block_size=None,
    verbose=False
):
    """
    Generalized Nyström method with optional block processing for RBF kernel.
    ...
    """
    rng = default_rng(random_state)

    # 1) Determine input type
    if kernel == "precomputed":
        K = X_or_K
        n = K.shape[0]
    elif kernel == "linear":
        n, p = X_or_K.shape
    elif kernel == "rbf":
        n, p = X_or_K.shape
        if gamma is None:
            raise ValueError("gamma must be specified for RBF kernel.")
    elif callable(kernel):
        n, p = X_or_K.shape
    else:
        shape = X_or_K.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            K = X_or_K
            n = K.shape[0]
            kernel = "precomputed"
        else:
            n, p = X_or_K.shape
            kernel = "linear"

    rank = _validate_rank(rank, n, "nystrom")

    # 2) Build C matrix
    C = np.zeros((n, rank), dtype=float)
    col_indices = rng.choice(n, size=rank, replace=False)

    # 3) Fill columns
    for idx_i, col_idx in enumerate(col_indices):
        if kernel == "precomputed":
            C[:, idx_i] = K[:, col_idx]
        elif kernel == "linear":
            C[:, idx_i] = X_or_K @ X_or_K[col_idx, :].T
        elif kernel == "rbf":
            col_data = X_or_K[col_idx, :]
            K_col = np.zeros(n, dtype=float)

            block_size = rbf_block_size if rbf_block_size else n
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                block = X_or_K[start:end, :]
                diffs = block - col_data
                sq_dists = np.sum(diffs**2, axis=1)
                K_col[start:end] = np.exp(-gamma * sq_dists)
                if verbose:
                    print(f"Processed block {start}-{end}")

            C[:, idx_i] = K_col
        else:  # callable
            for r in range(n):
                C[r, idx_i] = kernel(X_or_K, r, col_idx)

    # 4) W and pseudo-inverse
    W = C[col_indices, :]
    if regularization > 0.0:
        W += regularization * np.eye(rank)
    W_pinv = pinv(W)

    # 5) Return factorized or full
    return (C, W_pinv) if return_factorized else C @ W_pinv @ C.T
