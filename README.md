# MatrixSketcher

[![PyPI](https://img.shields.io/pypi/v/matrixsketcher?color=blue)](https://pypi.org/project/matrixsketcher/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/matrixsketcher.svg)](https://pypi.org/project/matrixsketcher/)
[![License](https://img.shields.io/github/license/luke-brosnan-cbc/matrixsketcher)](LICENSE)

A Python library for matrix sketching — randomised algorithms for dimensionality reduction, leverage score sampling, CUR decomposition, and structured transforms. The general aim is to approximate large matrices with compact representations that preserve the properties needed for downstream computation (inner products, low-rank structure, spectral information), at substantially reduced memory and runtime cost.

---

## Installation

```bash
pip install matrixsketcher
```

---

## Algorithms

### 1. CountSketch

A hashing-based compression method that maps $P$ features into $D \ll P$ buckets while approximately preserving inner products.

Given $X \in \mathbb{R}^{N \times P}$, define:
- A hash function $h: [1,P] \to [1,D]$ mapping each column to a sketch bucket
- A sign function $\sigma: [1,P] \to \{+1,-1\}$ assigning each column a random sign

The sketch matrix $S$ is constructed as:

$$S_{j,h(j)} = \sigma(j), \quad S_{j,k} = 0 \quad \text{for } k \neq h(j)$$

The compressed matrix is:

$$X' = X S^T, \quad X' \in \mathbb{R}^{N \times D}$$

CountSketch approximately preserves inner products: $\langle x_i, x_j \rangle \approx \langle x_i S^T, x_j S^T \rangle$, with expected reconstruction error bounded by:

$$\mathbb{E}\left[\|X - XS^TS\|_F^2\right] \leq \frac{1}{D}\|X\|_F^2$$

Increasing $D$ reduces error at the cost of a larger sketch.

**Typical applications:** feature hashing in NLP, dimensionality reduction for high-dimensional regression.

<details>
<summary>Example usage</summary>

```python
import numpy as np
from matrixsketcher.countsketch import countsketch

X = np.random.rand(1000, 5000)

X_sketch = countsketch(X, sketch_size=500, random_state=42)

print("Original shape:", X.shape)   # (1000, 5000)
print("Sketched shape:", X_sketch.shape)  # (1000, 500)
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `X` | array or sparse matrix | — | Input matrix of shape (n, p) |
| `sketch_size` | int | — | Target feature dimension $D$ |
| `random_state` | int | `None` | Random seed |
| `sparse_output` | bool | `True` | Build $S$ as sparse matrix; set `False` for dense |

</details>

---

### 2. Leverage Score Sampling

Samples rows or columns with probability proportional to their statistical importance in the SVD, rather than uniformly at random. Also supports uniform and norm-weighted sampling.

**Step 1:** Compute rank-$k$ SVD: $X \approx U_k S_k V_k^T$

**Step 2:** Compute leverage scores:

- Row sampling probability: $p_i^{(\text{row})} = \dfrac{\|U_{i,:}\|^2}{\sum_{i=1}^{N}\|U_{i,:}\|^2}$

- Column sampling probability: $p_j^{(\text{col})} = \dfrac{\|V_{:,j}\|^2}{\sum_{j=1}^{P}\|V_{:,j}\|^2}$

**Step 3:** Draw $D_{\text{row}}$ or $D_{\text{col}}$ indices according to the computed probabilities.

Weighted row sampling (by row norms) uses $p_i = \dfrac{\|X_{i,:}\|^2}{\sum_{r=1}^{N}\|X_{i,:}\|^2}$ as an alternative to SVD-based scores.

**Typical applications:** subsampling large datasets for econometric estimation, spectral methods, graph analytics.

<details>
<summary>Example usage</summary>

```python
import numpy as np
from matrixsketcher.leverage_score_sampling import leverage_score_sampling

X = np.random.rand(1000, 500)

X_sampled_rows = leverage_score_sampling(X, sample_size=100, sampling="leverage", axis=0, random_state=42)
X_sampled_cols = leverage_score_sampling(X, sample_size=50, sampling="leverage", axis=1, random_state=42)

print("Original shape:", X.shape)           # (1000, 500)
print("Sampled rows shape:", X_sampled_rows.shape)  # (100, 500)
print("Sampled columns shape:", X_sampled_cols.shape)  # (1000, 50)
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `X` | array or sparse matrix | — | Input matrix of shape (n, p) |
| `sample_size` | int | — | Number of rows (axis=0) or columns (axis=1) to sample |
| `rank` | int | `None` | SVD rank for leverage scores; defaults to min(n, p) |
| `random_state` | int | `None` | Random seed |
| `scale` | bool | `True` | Scale selected rows/columns by $1/\sqrt{p_i}$ for unbiased estimates |
| `sampling` | str | `"leverage"` | `"leverage"`, `"uniform"`, or `"weighted"` (rows only) |
| `axis` | int | `0` | `0` for row sampling, `1` for column sampling |

</details>

---

### 3. CUR Decomposition

CUR selects actual rows and columns from $X$ rather than forming abstract linear combinations, making the decomposition directly interpretable in terms of the original features and observations.

Given $X \in \mathbb{R}^{N \times K}$:

$$X \approx C W^{\dagger} R$$

where:
- $C \in \mathbb{R}^{N \times D_{\text{col}}}$ — selected columns of $X$
- $R \in \mathbb{R}^{D_{\text{row}} \times K}$ — selected rows of $X$
- $W \in \mathbb{R}^{D_{\text{row}} \times D_{\text{col}}}$ — submatrix at the intersection of selected rows and columns
- $W^{\dagger}$ — Moore-Penrose pseudoinverse of $W$

The pseudoinverse $W^{\dagger}$ corrects for correlations between the selected rows and columns. Without it, applying leverage score sampling independently to rows and columns does not account for the joint intersection structure, leading to a less accurate reconstruction.

Row selection supports uniform or leverage score sampling; columns are always selected by leverage scores.

**Typical applications:** feature selection, recommendation systems, identifying important factors in large econometric datasets.

<details>
<summary>Example usage</summary>

```python
import numpy as np
from matrixsketcher.cur_decomposition import cur_decomposition

X = np.random.rand(1000, 500)

C, W, R = cur_decomposition(X, d_rows=50, d_cols=40, rank=20, random_state=42, sampling="leverage")

print("C shape:", C.shape)  # (1000, 40)
print("W shape:", W.shape)  # (50, 40)
print("R shape:", R.shape)  # (50, 500)
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `X` | array or sparse matrix | — | Input matrix of shape (n, p) |
| `d_rows` | int | — | Number of rows to select |
| `d_cols` | int | — | Number of columns to select |
| `rank` | int | `None` | SVD rank for leverage scores; defaults to min(n, p) - 1 |
| `random_state` | int | `None` | Random seed |
| `sampling` | str | `"uniform"` | `"uniform"` or `"leverage"` for row selection |
| `regularization` | float | `0.0` | Diagonal regularisation added to $W$ before pseudoinversion |

</details>

---

### 4. Bicriteria CUR

An extension of CUR that jointly optimises row and column selection by iteratively swapping candidates to maximise the volume (determinant) of the intersection submatrix $W$. This provides a near-optimal reconstruction guarantee relative to the best rank-$k$ approximation.

The selection objective is:

$$(S_r^{\ast}, S_c^{\ast}) = \arg\max_{S_r, S_c} \det(W_{S_r, S_c}^T W_{S_r, S_c})$$

subject to $|S_r| = d_{\text{rows}}$, $\quad |S_c| = d_{\text{cols}}$.

Since exact maximisation is NP-hard, a greedy approach is used: initialise via leverage scores, then iteratively swap rows and columns to increase $\det(W^T W)$. This yields the provable bound:

$$\|X - CW^{\dagger}R\|_F \leq (1 + \epsilon)\|X - X_k\|_F$$

where $\epsilon$ depends on the number of selected rows and columns and $X_k$ is the best rank-$k$ approximation.

**Typical applications:** datasets with strong row-column dependence (e.g. gene expression, adjacency matrices), settings where standard CUR reconstruction quality is insufficient.

<details>
<summary>Example usage</summary>

```python
import numpy as np
from matrixsketcher.bicriteria_cur import bicriteria_cur

X = np.random.rand(1000, 500)

C, W, R = bicriteria_cur(X, d_rows=50, d_cols=40, rank=20, random_state=42, max_iter=6)

print("C shape:", C.shape)  # (1000, 40)
print("W shape:", W.shape)  # (50, 40)
print("R shape:", R.shape)  # (50, 500)
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `X` | array or sparse matrix | — | Input matrix of shape (n, p) |
| `d_rows` | int | — | Number of rows to select |
| `d_cols` | int | — | Number of columns to select |
| `rank` | int | `None` | SVD rank for leverage scores; defaults to min(n, p) - 1 |
| `random_state` | int | `None` | Random seed |
| `regularization` | float | `0.0` | Diagonal regularisation added to $W$ before pseudoinversion |
| `max_iter` | int | `6` | Greedy swap iterations; higher values improve selection at increased runtime |

</details>

---

### 5. Fast Walsh-Hadamard Transform (FWHT)

A structured, deterministic transform analogous to the FFT but operating over $\pm 1$ values rather than complex exponentials. The Walsh-Hadamard matrix is defined recursively:

$$H_1 = [1], \qquad H_{2n} =\begin{bmatrix} H_n & H_n \\\ H_n & -H_n \end{bmatrix}$$

For $X \in \mathbb{R}^{N \times P}$ with $N = 2^k$, the transform applies $H_N$ row-wise:

$$X' = H_N \cdot X$$

This runs in $O(N \log N)$ time and preserves inner products up to a scaling factor. If $N$ is not a power of 2, rows can be zero-padded to the next power of 2 via `pad_or_error="pad"`.

**Typical applications:** preconditioning for iterative solvers, randomised projections in regression, fast feature compression.

<details>
<summary>Example usage</summary>

```python
import numpy as np
from matrixsketcher.Fast_Walsh_Hadamard_Transform import fwht

X = np.random.rand(1024, 50)

X_fwht = fwht(X, pad_or_error="pad", random_state=42)

print("FWHT-transformed shape:", X_fwht.shape)  # (1024, 50)
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `X` | array or sparse matrix | — | Input matrix of shape (n, p); sparse is converted to dense |
| `random_state` | int | `None` | Random seed for sign randomisation |
| `pad_or_error` | str | `"error"` | `"error"` raises if $N$ is not a power of 2; `"pad"` zero-pads to next power of 2 |

</details>

---

## Contributing

Issues and pull requests welcome on [GitHub](https://github.com/luke-brosnan-cbc/matrixsketcher).

## License

See [LICENSE](LICENSE).
