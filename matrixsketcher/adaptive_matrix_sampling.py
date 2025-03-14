# matrixsketcher/adaptive_matrix_sampling.py


import numpy as np
from numpy.random import default_rng
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import isspmatrix


def adaptive_matrix_sampling(X, d_rows, d_cols, rank=None, max_iters=5, random_state=None):
    """
    Adaptive Matrix Sampling (AMM) that iteratively updates row and column importance scores.
    
    Parameters:
    - X (array or sparse matrix): Input data matrix (n x k)
    - d_rows (int): Number of rows to sample
    - d_cols (int): Number of columns to sample
    - rank (int, optional): Rank for SVD-based leverage score computation (default: min(n, p) - 1)
    - max_iters (int): Maximum number of iterative updates
    - random_state (int, optional): Seed for reproducibility

    Returns:
    - X_subset (array): A sampled subset of rows and columns from X.
    """
    rng = default_rng(random_state)
    n, p = X.shape

    if d_rows > n or d_cols > p:
        raise ValueError("Sample size cannot exceed matrix dimensions.")

    # Fix: Ensure rank is valid for svds()
    rank = min(rank or min(n, p) - 1, min(n, p) - 1)

    # Step 1: Compute leverage scores for rows from full X
    U, s, Vt = svds(X, k=rank) if isspmatrix(X) else svds(X, k=rank)
    row_leverage_scores = np.sum(U**2, axis=1)
    row_probs = row_leverage_scores / np.sum(row_leverage_scores)

    selected_rows = rng.choice(n, size=d_rows, replace=False, p=row_probs)

    for _ in range(max_iters):
        # Step 2: Extract selected rows from the original X
        X_sampled_rows = X[selected_rows, :]

        # Step 3: Compute leverage scores for columns based on the selected rows
        _, _, Vt = svds(X_sampled_rows, k=min(rank, min(X_sampled_rows.shape) - 1)) if isspmatrix(X_sampled_rows) else svds(X_sampled_rows, k=min(rank, min(X_sampled_rows.shape) - 1))
        col_leverage_scores = np.sum(Vt**2, axis=0)
        col_probs = col_leverage_scores / np.sum(col_leverage_scores)

        # Step 4: Sample columns from the full original X
        selected_cols = rng.choice(p, size=d_cols, replace=False, p=col_probs)

    # Step 5: Return the subset from the full original X
    return X[np.ix_(selected_rows, selected_cols)]
