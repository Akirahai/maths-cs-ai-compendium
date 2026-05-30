# Exercises — Section 7: Eigenvalues, SVD, and PCA

> **Difficulty key**: ★ warm-up · ★★ core · ★★★ applied ML · ★★★★ challenge

---

## Conceptual Questions

**C1. ★** An eigenvector stays on the same line after being multiplied by a matrix. What happens to a general (non-eigenvector) direction? Sketch a $2\times2$ example showing a matrix that rotates most vectors but leaves one direction fixed.

**C2. ★★** The trace of a matrix equals the sum of its eigenvalues. Verify this for $A = \begin{bmatrix}3 & 1\\0 & 2\end{bmatrix}$ without computing the eigenvectors. What does this say about the relationship between diagonal entries and eigenvalues for triangular matrices?

**C3. ★★** PCA finds the directions of maximum variance. Why does this correspond to the eigenvectors of the covariance matrix (and not, say, the mean-deviation matrix directly)?

**C4. ★★★** SVD works for any matrix, but eigendecomposition only works for square matrices. For a non-square matrix $A$ ($m \times n$, $m > n$):
- The left singular vectors (columns of $U$) are eigenvectors of $AA^T$
- The right singular vectors (columns of $V$) are eigenvectors of $A^TA$

How does this connect SVD to PCA? (Hint: $\tilde{X}^T\tilde{X}$ is proportional to the covariance matrix.)

---

## Pencil-and-Paper Problems

**P1. ★** Find the eigenvalues and eigenvectors of:

(a) $A = \begin{bmatrix}2 & 0\\0 & 3\end{bmatrix}$ (diagonal — read them off directly)

(b) $A = \begin{bmatrix}0 & 1\\1 & 0\end{bmatrix}$ (permutation matrix)

(c) $A = \begin{bmatrix}1 & 2\\0 & 1\end{bmatrix}$ (shear — careful with repeated eigenvalue)

(d) $A = \begin{bmatrix}3 & 1\\1 & 3\end{bmatrix}$ (symmetric — verify eigenvectors are orthogonal)

**P2. ★★** For $A = \begin{bmatrix}4 & 2\\1 & 3\end{bmatrix}$:

(a) Find the characteristic polynomial $\det(A - \lambda I)$.  
(b) Find the eigenvalues.  
(c) Find the eigenvectors.  
(d) Write the eigendecomposition $A = PDP^{-1}$.  
(e) Use it to compute $A^3$ efficiently (compute $PD^3P^{-1}$).

**P3. ★★** Perform PCA by hand on these 4 points in $\mathbb{R}^2$: $(1,2), (2,4), (3,5), (4,7)$.

(a) Compute the mean and mean-centre the data.  
(b) Compute the $2\times2$ covariance matrix.  
(c) Find the eigenvalues and eigenvectors.  
(d) What percentage of variance does the first PC explain?  
(e) Project all 4 points onto the first PC. What do you notice about the projected values?

**P4. ★★** The SVD of a matrix $A$ is $A = U\Sigma V^T$.

For $A = \begin{bmatrix}1 & 1\\0 & 1\\1 & 0\end{bmatrix}$:

(a) Compute $A^TA$ (a $2\times2$ matrix). Find its eigenvalues — these are $\sigma_1^2, \sigma_2^2$.  
(b) Find the singular values $\sigma_1 \geq \sigma_2$.  
(c) Find $V$ (eigenvectors of $A^TA$) and $U$ (compute $\mathbf{u}_k = \frac{1}{\sigma_k}A\mathbf{v}_k$).  
(d) Verify $A = U\Sigma V^T$.

**P5. ★★★** **Low-rank approximation**. The matrix:

$$A = \begin{bmatrix}1 & 2\\3 & 6\end{bmatrix}$$

(a) Compute the rank of $A$.  
(b) Compute the SVD. How many nonzero singular values are there?  
(c) Write the rank-1 approximation $A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$.  
(d) This should equal $A$ exactly — why?

**P6. ★★★** **Power iteration**. The dominant eigenvector can be found by repeatedly multiplying and normalising.

For $A = \begin{bmatrix}3 & 1\\1 & 2\end{bmatrix}$, starting from $\mathbf{v}^{(0)} = [1, 0]^T$:

(a) Compute $\mathbf{v}^{(1)} = A\mathbf{v}^{(0)} / \|A\mathbf{v}^{(0)}\|$.  
(b) Compute $\mathbf{v}^{(2)} = A\mathbf{v}^{(1)} / \|A\mathbf{v}^{(1)}\|$.  
(c) Continue until convergence. Compare to the true dominant eigenvector.  
(d) The Rayleigh quotient $r(\mathbf{v}) = \mathbf{v}^TA\mathbf{v}/\mathbf{v}^T\mathbf{v}$ estimates the eigenvalue. Compute it at each step and observe convergence.

---

## Applied ML Problems

**A1. ★★** **Scree plot and choosing $k$ (Week 9)**. A dataset has covariance matrix with eigenvalues:

$$[15.2,\; 8.1,\; 3.4,\; 2.9,\; 1.1,\; 0.8,\; 0.3,\; 0.2]$$

(a) Compute the total variance (sum of eigenvalues).  
(b) Compute the cumulative variance explained for $k = 1, 2, \ldots, 8$.  
(c) How many principal components do you need to retain 90% of the variance?  
(d) The "elbow" of the scree plot (where the eigenvalues drop steeply then level off) is a common heuristic. Where is the elbow here?

**A2. ★★** **PCA for compression**. A greyscale image can be represented as a $64 \times 64$ matrix (4096 pixels). Treating each row (64 pixels) as a sample:

(a) What are the dimensions of the data matrix $X$?  
(b) After PCA, you keep the top $k = 10$ principal components. What is the compressed representation size (in numbers)? What is the compression ratio?  
(c) The reconstruction is $\hat{X} = Z W_k^T + \bar{X}$ where $Z = \tilde{X}W_k$. Write out all shapes.  
(d) What is the reconstruction error in terms of the discarded eigenvalues?

**A3. ★★★** **Linear autoencoder = PCA**. A linear autoencoder has:
- Encoder: $\mathbf{z} = W_e\mathbf{x}$ (maps $\mathbb{R}^n \to \mathbb{R}^k$)
- Decoder: $\hat{\mathbf{x}} = W_d\mathbf{z}$ (maps $\mathbb{R}^k \to \mathbb{R}^n$)

Trained to minimise $\|X - XW_e^TW_d^T\|_F^2$.

(a) Show that the optimal $W_d = W_e^T$ (the decoder is the transpose of the encoder) when weights are orthonormal.  
(b) The optimal encoder rows are the top $k$ eigenvectors of $X^TX$. Why does this mean the autoencoder learns the same subspace as PCA?  
(c) What can a nonlinear autoencoder (with ReLU activations) do that PCA cannot?

**A4. ★★★** **HMM stationary distribution (Week 10)**. A Markov chain has transition matrix:

$$T = \begin{bmatrix}0.7 & 0.3\\0.4 & 0.6\end{bmatrix}$$

where $T_{ij} = P(\text{state}=j \mid \text{state}=i)$.

(a) Find the eigenvalues of $T$.  
(b) The stationary distribution $\boldsymbol{\pi}$ satisfies $\boldsymbol{\pi}^T T = \boldsymbol{\pi}^T$ (it is a left eigenvector with eigenvalue 1). Find $\boldsymbol{\pi}$ by solving $\boldsymbol{\pi}^T(T - I) = 0$ with $\pi_1 + \pi_2 = 1$.  
(c) Verify: starting from any initial distribution $\boldsymbol{\pi}^{(0)} = [1, 0]$ (always in state 1), compute $\boldsymbol{\pi}^{(t)} = \boldsymbol{\pi}^{(0)} T^t$ for $t = 1, 2, 5, 10$. Does it converge to $\boldsymbol{\pi}$?  
(d) How does the second eigenvalue determine the rate of convergence?

---

## Coding Exercises

**Code 1. ★** Compute eigendecomposition and verify properties:

```python
import jax.numpy as jnp

A = jnp.array([[4.0, 2.0],
               [2.0, 3.0]])

# Eigendecomposition (eigh for symmetric matrices)
eigenvalues, eigenvectors = jnp.linalg.eigh(A)
print(f"Eigenvalues: {eigenvalues}")

# Verify 1: A v = λ v for each eigenpair
for i in range(2):
    lam, v = eigenvalues[i], eigenvectors[:, i]
    Av = A @ v
    lv = lam * v
    print(f"λ={lam:.4f}: ||Av - λv|| = {jnp.linalg.norm(Av - lv):.2e}")

# Verify 2: trace = sum of eigenvalues
print(f"tr(A)={jnp.trace(A):.4f}, sum(λ)={eigenvalues.sum():.4f}")

# Verify 3: det = product of eigenvalues
print(f"det(A)={jnp.linalg.det(A):.4f}, prod(λ)={eigenvalues.prod():.4f}")

# Verify 4: reconstruction A = P D P^T
P = eigenvectors
D = jnp.diag(eigenvalues)
A_reconstructed = P @ D @ P.T
print(f"Reconstruction error: {jnp.linalg.norm(A - A_reconstructed):.2e}")
```

**Code 2. ★★** Full PCA pipeline with visualisation:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Generate correlated 2D data
rng = np.random.default_rng(1)
angle = np.pi / 6
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])
X_raw = jnp.array((R @ rng.normal(size=(2, 200)) * np.array([[3], [1]])).T)

# PCA from scratch
def pca(X, k):
    """Return projected data Z, eigenvectors W, eigenvalues, explained variance ratio."""
    X_c = X - X.mean(axis=0)
    m = X.shape[0]
    Cov = (X_c.T @ X_c) / (m - 1)
    vals, vecs = jnp.linalg.eigh(Cov)
    # Sort descending
    idx = jnp.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    W_k = vecs[:, :k]
    Z = X_c @ W_k
    explained = vals / vals.sum()
    return Z, W_k, vals, explained

Z, W, vals, explained = pca(X_raw, k=1)
print(f"Eigenvalues: {vals}")
print(f"Variance explained by PC1: {explained[0]*100:.1f}%")

# Reconstruct
X_c = X_raw - X_raw.mean(axis=0)
X_recon = Z @ W.T + X_raw.mean(axis=0)
recon_error = jnp.linalg.norm(X_raw - X_recon, ord='fro')
print(f"Reconstruction error (Frobenius): {recon_error:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X_raw[:, 0], X_raw[:, 1], alpha=0.5, s=10, label='original')
ax.scatter(X_recon[:, 0], X_recon[:, 1], alpha=0.5, s=10, c='red', label='projected (k=1)')
origin = X_raw.mean(axis=0)
for i, (v, name) in enumerate(zip(W.T, ['PC1'])):
    ax.quiver(*origin, *v * vals[i]**0.5 * 2, scale=1, scale_units='xy',
              color=['blue', 'green'][i], label=name, width=0.01)
ax.set_aspect('equal'); ax.legend(); ax.grid(True)
plt.title(f"PCA: PC1 explains {explained[0]*100:.1f}% variance")
plt.tight_layout(); plt.show()
```

**Code 3. ★★** Image compression with SVD:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

# Load a single digit image (8x8 pixels)
digits = load_digits()
img = jnp.array(digits.images[0])   # shape (8, 8)

# SVD
U, S, Vt = jnp.linalg.svd(img, full_matrices=False)
print(f"Singular values: {S.round(2)}")

fig, axes = plt.subplots(1, 5, figsize=(14, 3))
axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original")

for idx, k in enumerate([1, 2, 4, 8]):
    img_k = U[:, :k] @ jnp.diag(S[:k]) @ Vt[:k, :]
    err = jnp.linalg.norm(img - img_k, ord='fro') / jnp.linalg.norm(img, ord='fro')
    axes[idx+1].imshow(img_k, cmap='gray')
    axes[idx+1].set_title(f"k={k}\nerr={err:.2%}")

plt.tight_layout(); plt.show()

# Plot explained variance
total = jnp.sum(S**2)
explained = jnp.cumsum(S**2) / total
plt.figure(figsize=(5, 3))
plt.plot(range(1, len(S)+1), explained * 100, 'o-')
plt.xlabel("Number of singular values k")
plt.ylabel("Cumulative variance explained (%)")
plt.title("Scree plot (SVD)"); plt.grid(True)
plt.tight_layout(); plt.show()
```

**Code 4. ★★★** Apply PCA to a real dataset and compare to an autoencoder:

```python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X_raw = jnp.array(StandardScaler().fit_transform(digits.data))   # (1797, 64)
y = digits.target

# --- PCA ---
X_c = X_raw - X_raw.mean(axis=0)
m = X_c.shape[0]
Cov = (X_c.T @ X_c) / (m - 1)
vals, vecs = jnp.linalg.eigh(Cov)
idx = jnp.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]

# Scree plot
explained = jnp.cumsum(vals) / vals.sum()
k90 = int(jnp.argmax(explained >= 0.90)) + 1
print(f"Components needed for 90% variance: {k90}")

# Project to 2D
Z2 = X_c @ vecs[:, :2]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(Z2[:, 0], Z2[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
axes[0].set_title("PCA: 2 components"); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

axes[1].plot(range(1, 21), explained[:20] * 100, 'o-')
axes[1].axvline(k90, color='red', linestyle='--', label=f'90% at k={k90}')
axes[1].set_xlabel("k"); axes[1].set_ylabel("Cumulative variance (%)")
axes[1].set_title("Scree curve — digits dataset"); axes[1].legend(); axes[1].grid(True)

plt.tight_layout(); plt.show()
```

---

## Solutions

<details>
<summary><strong>P1(d) Solution</strong></summary>

$A = \begin{bmatrix}3&1\\1&3\end{bmatrix}$, characteristic polynomial: $(3-\lambda)^2 - 1 = 0 \Rightarrow \lambda^2 - 6\lambda + 8 = 0 \Rightarrow \lambda = 2, 4$.

For $\lambda_1 = 2$: $(A-2I)\mathbf{v} = 0 \Rightarrow \begin{bmatrix}1&1\\1&1\end{bmatrix}\mathbf{v} = 0 \Rightarrow \mathbf{v}_1 = [1,-1]^T/\sqrt{2}$

For $\lambda_2 = 4$: $(A-4I)\mathbf{v} = 0 \Rightarrow \begin{bmatrix}-1&1\\1&-1\end{bmatrix}\mathbf{v} = 0 \Rightarrow \mathbf{v}_2 = [1,1]^T/\sqrt{2}$

Check orthogonality: $\mathbf{v}_1 \cdot \mathbf{v}_2 = (1)(1) + (-1)(1) = 0$ ✓ (symmetric matrices always have orthogonal eigenvectors)

</details>

<details>
<summary><strong>P3 Solution</strong></summary>

Mean: $\bar{\mathbf{x}} = [2.5, 4.5]^T$

Mean-centred: $[(-1.5,-2.5), (-0.5,-0.5), (0.5,0.5), (1.5,2.5)]$

$\Sigma = \frac{1}{3}\begin{bmatrix}(-1.5)^2+\ldots & (-1.5)(-2.5)+\ldots \\ \ldots & (-2.5)^2+\ldots\end{bmatrix} = \begin{bmatrix}5/3 & 8/3\\8/3 & 35/6\end{bmatrix}$

Wait — let me redo: $\tilde{X}^T\tilde{X} = \begin{bmatrix}2.25+0.25+0.25+2.25 & 3.75+0.25+0.25+3.75\\ 3.75+0.25+0.25+3.75 & 6.25+0.25+0.25+6.25\end{bmatrix} = \begin{bmatrix}5 & 8\\8 & 13\end{bmatrix}$

$\Sigma = \frac{1}{3}\begin{bmatrix}5&8\\8&13\end{bmatrix}$

Characteristic polynomial: $\lambda^2 - 6\lambda + (65-64)/9 = 0$... The data is nearly collinear (lies almost on a line), so one eigenvalue dominates. PC1 explains ~99% of variance.

</details>

<details>
<summary><strong>A1 Solution</strong></summary>

(a) Total variance $= 15.2+8.1+3.4+2.9+1.1+0.8+0.3+0.2 = 32.0$

(b) Cumulative: $[15.2, 23.3, 26.7, 29.6, 30.7, 31.5, 31.8, 32.0]$ → as percentages: $[47.5\%, 72.8\%, 83.4\%, 92.5\%, 95.9\%, 98.4\%, 99.4\%, 100\%]$

(c) $k=4$ retains $92.5\% > 90\%$

(d) Elbow is between $k=2$ and $k=3$ — the eigenvalues drop sharply from 8.1 to 3.4 then gradually after.

</details>
