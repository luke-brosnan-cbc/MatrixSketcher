# matrixsketcher/bicriteria_cur.py


import numpy as np
from numpy.random import default_rng
from scipy.linalg import svd, pinv
from scipy.sparse.linalg import svds
from scipy.sparse import isspmatrix


def bicriteria_cur(X, d_rows, d_cols, rank=None, random_state=None):
    """
    Bicriteria CUR: Joint optimization of row and column selection. 
    Based on Boutsidis et al. (2014).
    
    Parameters:
    - X (array or sparse matrix): Input data matrix (n x k)
    - d_rows (int): Number of rows to select
    - d_cols (int): Number of columns to select
    - rank (int, optional): Rank for leverage score computation (default: min(n, p) - 1)
    - random_state (int, optional): Seed for reproducibility

    Returns:
    - X_approx (array): CUR approximation of X
    """
    rng = default_rng(random_state)
    n, p = X.shape

    if d_rows > n or d_cols > p:
        raise ValueError("Sample size cannot exceed matrix dimensions.")

    # Ensure rank is always valid
    rank = min(rank or min(n, p) - 1, min(n, p) - 1)

    # Step 1: Compute row and column leverage scores from SVD
    U, s, Vt = svds(X, k=rank) if isspmatrix(X) else svds(X, k=rank)

    row_scores = np.sum(U**2, axis=1)
    col_scores = np.sum(Vt**2, axis=0)

    row_probs = row_scores / np.sum(row_scores)
    col_probs = col_scores / np.sum(col_scores)

    # Step 2: Joint sampling of rows and columns
    selected_rows = rng.choice(n, size=d_rows, replace=False, p=row_probs)
    selected_cols = rng.choice(p, size=d_cols, replace=False, p=col_probs)

    # Step 3: Extract the CUR components
    C = X[:, selected_cols]  # Select columns
    R = X[selected_rows, :]  # Select rows
    W = X[np.ix_(selected_rows, selected_cols)]  # Intersection matrix

    # Step 4: Compute pseudoinverse of W for CUR reconstruction
    W_pinv = pinv(W)

    # Step 5: Compute CUR approximation
    X_approx = C @ W_pinv @ R  # Matrix multiplication to reconstruct X

    return X_approx
