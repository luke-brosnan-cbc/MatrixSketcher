# SketchLib: Efficient Matrix Sketching for Large-Scale Computations

[![PyPI](https://img.shields.io/pypi/v/sketchlib?color=blue)](https://pypi.org/project/sketchlib/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sketchlib)](https://pypi.org/project/sketchlib/)
[![License](https://img.shields.io/github/license/luke-brosnan-cbc/sketchlib)](LICENSE)

SketchLib is a high-performance Python library for **matrix sketching**, enabling scalable and memory-efficient approximations for large matrices. It provides a suite of randomized algorithms for **dimensionality reduction, kernel approximation, leverage score sampling, and compressed linear algebra.**

---

## 🚀 **What is Matrix Sketching? Why is it Useful?**
Matrix sketching is a technique used to **approximate large matrices with a much smaller representation**, making computations significantly **faster and more memory-efficient**.

Instead of processing an entire large dataset, matrix sketching allows you to:
- **Reduce storage requirements** by keeping only a compressed form of the data.
- **Speed up machine learning and econometrics models** without losing key information.
- **Approximate costly transformations** like covariance matrices and kernel functions.

### 🔥 **Where is Matrix Sketching Used?**
- **Machine Learning (ML)**: Speeding up PCA, kernel methods, and regression models.
- **Econometrics & Finance**: Handling massive datasets efficiently in regressions and covariance estimation.
- **Natural Language Processing (NLP)**: Compressing large word embedding matrices.
- **Graph & Network Analysis**: Speeding up computations on **social networks, blockchain transactions, and recommendation systems**.

---

## 🏗 **How Do Different Sketching Methods Differ?**
Each method in SketchLib serves a different purpose:
- **Random Projection**: Reduces dimensions while preserving distances.
- **Subsampled SVD**: Creates a low-rank summary of a matrix.
- **Nyström Approximation**: Speeds up kernel-based methods.
- **CUR Decomposition**: Selects **actual rows and columns** for interpretability.
- **CountSketch**: Compresses matrices using hashing techniques.
- **Leverage Score Sampling**: Smart sampling that keeps **important** data points.
- **Fast Walsh-Hadamard Transform (FWHT)**: Structured projections for efficient compression.

---

## 🔢 **Core Algorithms & Real-World Use Cases**

### **1. Random Projection (Johnson-Lindenstrauss)**
💡 **Best for:** **Dimensionality reduction**, speeding up ML models, feature compression.

**🔹 What it does:**  
Random Projection reduces the number of features **while preserving pairwise distances** between data points. This is crucial for ML applications where high-dimensional data slows down computation.

📌 **Example Use Cases:**
- **Speeding up nearest neighbor search** in recommendation systems.
- **Reducing computational cost in large-scale regressions**.
- **Making high-dimensional econometric models more efficient**.

📏 **Mathematical Formulation:**
<div align="center"; margin: 0>
  
### $X' = X R$

</div>

where:
- $X \in \mathbb{R}^{n \times p}$ is the original dataset.
- $R \in \mathbb{R}^{p \times d}$ is a random matrix.
- $X' \in \mathbb{R}^{n \times d}$ is the lower-dimensional sketch.

---

### **2. Subsampled Singular Value Decomposition (SVD)**
💡 **Best for:** **Finding patterns in data, PCA, recommendation systems**.

**🔹 What it does:**  
SVD decomposes a dataset into **simpler components**, but full computation is expensive. Subsampled SVD picks a **small subset of rows** and computes a **low-rank approximation**, making it **much faster**.

📌 **Example Use Cases:**
- **Efficient PCA for high-dimensional data**.
- **Faster matrix factorization in recommendation engines**.
- **Summarizing trends in econometric datasets**.

📏 **Mathematical Formulation:**
<div align="center"; margin: 0>
  
### $X' = U S V^T$

</div>

where:
- $U, S, V$ are derived from a **subsampled** version of $X$.

---

### **3. Nyström Approximation (Fast Kernel Methods)**
💡 **Best for:** **Speeding up kernel-based ML models (SVMs, Gaussian Processes, Spectral Clustering)**.

**🔹 What it does:**  
Kernel methods (like SVMs and Gaussian Processes) use large **similarity matrices**, which scale poorly. Nyström approximation **selects only a subset of columns**, greatly speeding up computation.

📌 **Example Use Cases:**
- **Scaling up kernel SVMs and Gaussian Processes**.
- **Fast spectral clustering for large datasets**.
- **Econometric covariance estimation for large asset portfolios**.

📏 **Mathematical Formulation:**
<div align="center"; margin: 0>
  
### $K \approx C W^{-1} C^T$

</div>

where:
- $K$ is the full kernel matrix.
- $C$ is a subset of columns.
- $W$ is a small **intersection matrix**.

---

### **4. CUR Decomposition (Interpretable Low-Rank Approximation)**
💡 **Best for:** **Feature selection, interpretability, compressed storage**.

**🔹 What it does:**  
CUR selects **actual rows and columns** instead of abstract components (like SVD), making results **more interpretable**.

📌 **Example Use Cases:**
- **Identifying the most important features** in large datasets.
- **Compressing massive recommendation matrices**.
- **Enhancing interpretability in econometric models**.

📏 **Mathematical Formulation:**
<div align="center"; margin: 0>
  
### $X \approx C W^{-1} R$

</div>

where:
- $C, R$ are selected rows and columns.
- $W$ is a core submatrix.

---

### **5. CountSketch (Feature Hashing)**
💡 **Best for:** **Reducing feature matrix size while preserving inner products**.

**🔹 What it does:**  
CountSketch uses a **randomized hashing technique** to efficiently project large matrices into a smaller space while retaining key information.

📌 **Example Use Cases:**
- **Reducing dimensionality in NLP models (e.g., word embeddings)**.
- **Fast feature engineering for large-scale ML and econometrics**.

📏 **Mathematical Formulation:**
<div align="center"; margin: 0>
  
### $X' = X S^T$

</div>

where $S$ is a **sparse, sign-randomized matrix**.

---

### **6. Leverage Score Sampling**
💡 **Best for:** **Choosing the most "informative" rows in a dataset**.

**🔹 What it does:**  
Instead of randomly picking rows, **Leverage Score Sampling** selects rows **proportionally to their statistical importance**.

📌 **Example Use Cases:**
- **Efficient econometric model estimation** with fewer samples.
- **Speeding up spectral clustering and graph-based ML**.

📏 **Mathematical Formulation:**
<div align="center"; margin: 0>
  
### $p_i = \frac{\sum_j U_{ij}^2}{\sum_{i,j} U_{ij}^2}$

</div>

where $U$ is the left singular matrix from SVD.

---

### **7. Fast Transforms (FWHT & FFT)**
💡 **Best for:** **Structured random projections, fast transforms in signal processing and machine learning**.

#### 🔹 **Fast Walsh-Hadamard Transform (FWHT)**
FWHT is a **structured random transformation** that replaces **dense random matrices** with a deterministic transform, making it computationally efficient.

📏 **Mathematical Formulation:**
<div align="center"; margin: 0>
  
$$H_{n} x = \frac{1}{\sqrt{n}} \left( \begin{bmatrix} 1 & 1 \\\ 1 & -1 \end{bmatrix} \right)^{\otimes \log_{2} n} x$$

</div>

**Where:**
- $H_n$ is the **Hadamard matrix**, which follows a recursive structure:
  
$$H_{2n} = \begin{bmatrix} H_n & H_n \\\ H_n & -H_n \end{bmatrix}$$

- $x$ is the input vector (or matrix).
- $n$ is the size of the transformation (must be a power of 2).
- The notation $\otimes \log_{2} n$ represents the **Kronecker power**, recursively expanding the transformation.

📌 **Example Use Cases:**
- **Speeding up least squares regression in ML**.
- **Preconditioning large econometric models**.

#### 🔹 **Fast Fourier Transform (FFT)**
FFT is a **widely used transformation** for analyzing frequency components in signals. Unlike FWHT, which uses **binary operations**, FFT is optimized for **sinusoids and continuous data**.

📏 Mathematical Formulation:

<div align="center"; margin: 0>

### $X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N}, \quad k = 0, \dots, N-1$

</div>

**Where:**
- $X_k$ represents the **Fourier coefficients**, capturing different frequency components.
- $x_n$ is the input signal in the **time domain**.
- $N$ is the total number of points in the signal.
- $e^{-2\pi i k n / N}$ is the complex exponential term, which represents rotations in the **frequency domain**.

📌 **Example Use Cases:**
- **Efficient spectral analysis in signal processing**.
- **Time series forecasting in econometrics**.
- **Speeding up convolutional operations in ML**.


**Key Differences:**

| **Feature**  | **FWHT** | **FFT**  |
|-------------|---------|---------|
| Works on  | Binary data | Continuous data (sinusoids) |
| Application | ML, econometrics | Signal processing, ML |
| Speed | Faster for structured matrices | Efficient for time series |



---
## 🔧 **Installation**
To install SketchLib, simply run:

```sh
pip install sketchlib
