# Exercises — Section 6: Covariance and Positive Definiteness

> **Difficulty key**: ★ warm-up · ★★ core · ★★★ applied ML · ★★★★ challenge

---

## Conceptual Questions

**C1. ★** What does a large off-diagonal entry $\Sigma_{12}$ in a covariance matrix tell you about features 1 and 2? How is this information used in a GMM vs. Naive Bayes?

**C2. ★★** A covariance matrix must be positive semi-definite (PSD). Give an intuitive reason why: if $\Sigma$ had a negative eigenvalue, what would that mean about the variance of the data in the corresponding direction?

**C3. ★★** Naive Bayes assumes features are conditionally independent given the class. In matrix language, this means $\Sigma$ is diagonal. What are the practical benefits of this assumption? What are the costs?

**C4. ★★** Euclidean distance treats all features equally. Mahalanobis distance $d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ accounts for scale and correlation. Give an example where Euclidean distance gives a misleading similarity judgment that Mahalanobis corrects.

---

## Pencil-and-Paper Problems

**P1. ★** Compute the covariance matrix for the following dataset (3 samples, 2 features):

$$X = \begin{bmatrix}1 & 4\\3 & 2\\5 & 6\end{bmatrix}$$

(a) Compute the mean vector $\boldsymbol{\mu}$.  
(b) Mean-centre the data: $\tilde{X} = X - \boldsymbol{\mu}^T$ (subtract row-wise).  
(c) Compute $\Sigma = \frac{1}{m-1}\tilde{X}^T\tilde{X}$.  
(d) Interpret each entry: what is $\text{Var}(x_1)$, $\text{Var}(x_2)$, and $\text{Cov}(x_1, x_2)$? Are the features positively or negatively correlated?

**P2. ★** For each matrix, determine whether it is positive definite (PD), positive semi-definite (PSD but not PD), or neither:

(a) $\begin{bmatrix}2 & 0\\0 & 3\end{bmatrix}$

(b) $\begin{bmatrix}1 & 2\\2 & 1\end{bmatrix}$

(c) $\begin{bmatrix}4 & 2\\2 & 1\end{bmatrix}$

(d) $\begin{bmatrix}3 & 1\\1 & 3\end{bmatrix}$

For each, compute the eigenvalues to verify your answer.

**P3. ★★** Compute the quadratic form $\mathbf{v}^T\Sigma\mathbf{v}$ for $\Sigma = \begin{bmatrix}4 & 2\\2 & 3\end{bmatrix}$ and each direction:

(a) $\mathbf{v} = [1, 0]^T$ (horizontal direction)  
(b) $\mathbf{v} = [0, 1]^T$ (vertical direction)  
(c) $\mathbf{v} = [1/\sqrt{2}, 1/\sqrt{2}]^T$ (45° diagonal)  
(d) $\mathbf{v} = [1/\sqrt{2}, -1/\sqrt{2}]^T$ (135° diagonal)

Interpret: in which direction is the data most spread out? Least spread out?

**P4. ★★** Compute Mahalanobis and Euclidean distances for the following setup:

Distribution mean $\boldsymbol{\mu} = [0, 0]^T$, covariance $\Sigma = \begin{bmatrix}9 & 0\\0 & 1\end{bmatrix}$.

Points: $\mathbf{p}_1 = (3, 0)$, $\mathbf{p}_2 = (0, 1)$, $\mathbf{p}_3 = (1, 1)$.

(a) Compute Euclidean distance from $\boldsymbol{\mu}$ for each point.  
(b) Compute Mahalanobis distance from $\boldsymbol{\mu}$ for each point.  
(c) According to Euclidean distance, $\mathbf{p}_1$ is further from $\boldsymbol{\mu}$ than $\mathbf{p}_2$. But according to Mahalanobis distance, which is "more surprising"? Explain why.

**P5. ★★** The **correlation matrix** is the covariance matrix after normalising each feature to unit variance:

$$\rho_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}$$

For $\Sigma = \begin{bmatrix}4 & 3\\3 & 9\end{bmatrix}$:

(a) Compute the correlation matrix $R$.  
(b) What is the range of each entry in $R$? What does $\rho_{12} = 1$ mean? $\rho_{12} = 0$?  
(c) A classifier uses raw features (heights in cm, weights in kg, ages in years). Why should you normalise before computing the covariance matrix?

---

## Applied ML Problems

**A1. ★★** **Gaussian Naive Bayes (Week 6)**. A binary classifier has two classes with the following statistics:

- Class $+1$: $\boldsymbol{\mu}_+ = [2, 3]$, $\Sigma_+ = \begin{bmatrix}1 & 0\\0 & 4\end{bmatrix}$
- Class $-1$: $\boldsymbol{\mu}_- = [-1, -1]$, $\Sigma_- = \begin{bmatrix}2 & 0\\0 & 1\end{bmatrix}$

(a) Both covariance matrices are diagonal. What does Naive Bayes assume about the features?  
(b) For a new point $\mathbf{x} = [1, 2]^T$, compute the Mahalanobis distance to each class mean.  
(c) Assuming equal class priors, which class does Naive Bayes predict? (Predict the class with smaller Mahalanobis distance.)  
(d) What would the full-covariance GMM assign? Would the result be the same?

**A2. ★★** **GMM M-step (Week 6)**. After the E-step, soft assignments (responsibilities) $r_{ik}$ are given:

| Point | $\mathbf{x}_i$ | $r_{i1}$ | $r_{i2}$ |
|-------|--------------|--------|--------|
| 1 | $(1, 1)$ | 0.9 | 0.1 |
| 2 | $(2, 1)$ | 0.8 | 0.2 |
| 3 | $(5, 4)$ | 0.1 | 0.9 |
| 4 | $(6, 5)$ | 0.1 | 0.9 |

For component $k=1$:

(a) Compute the effective count $N_1 = \sum_i r_{i1}$.  
(b) Compute the updated mean $\boldsymbol{\mu}_1 = \frac{1}{N_1}\sum_i r_{i1}\mathbf{x}_i$.  
(c) Compute the updated covariance $\Sigma_1 = \frac{1}{N_1}\sum_i r_{i1}(\mathbf{x}_i - \boldsymbol{\mu}_1)(\mathbf{x}_i - \boldsymbol{\mu}_1)^T$.  
(d) Is $\Sigma_1$ positive definite? Check the eigenvalues.

**A3. ★★★** **Quadratic discriminant analysis**. The QDA decision boundary between class $+1$ and class $-1$ is the set of points where the log-likelihood ratio is zero:

$$-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_+)^T\Sigma_+^{-1}(\mathbf{x}-\boldsymbol{\mu}_+) + \frac{1}{2}\ln|\Sigma_+| = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_-)^T\Sigma_-^{-1}(\mathbf{x}-\boldsymbol{\mu}_-) + \frac{1}{2}\ln|\Sigma_-|$$

For the Gaussian Naive Bayes in A1:

(a) What is the shape of this boundary in general (it is a quadric surface — what 2D shape)?  
(b) If $\Sigma_+ = \Sigma_-$ (same covariance for both classes), show that the boundary simplifies to a **linear** function of $\mathbf{x}$. This is **Linear Discriminant Analysis (LDA)**.  
(c) Why does LDA need fewer parameters than QDA? When would you prefer QDA?

**A4. ★★★★** **Covariance estimation with limited data**. You have $m$ samples in $\mathbb{R}^n$.

(a) How many independent parameters does a full covariance matrix have? (Answer: $n(n+1)/2$.)  
(b) For the sample covariance $\hat{\Sigma} = \frac{1}{m-1}\tilde{X}^T\tilde{X}$ to be invertible, you need $m > n$. Why?  
(c) **Ledoit-Wolf shrinkage** regularises: $\hat{\Sigma}_{\text{LW}} = (1-\alpha)\hat{\Sigma} + \alpha \frac{\text{tr}(\hat{\Sigma})}{n}I$. What does this interpolate between? Why does the identity covariance represent the "most uncertain" prior?  
(d) For $n = 100$ features and $m = 50$ samples, the sample covariance is rank-deficient. What rank will it have? What does this imply for GMM fitting?

---

## Coding Exercises

**Code 1. ★★** Visualise the Mahalanobis distance as ellipses:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Three covariance matrices (same mean, different shapes)
mu = jnp.array([0.0, 0.0])
covariances = {
    "Isotropic (σ²I)":    jnp.array([[2.0, 0.0], [0.0, 2.0]]),
    "Diagonal (unequal)": jnp.array([[4.0, 0.0], [0.0, 1.0]]),
    "Full (correlated)":  jnp.array([[3.0, 2.0], [2.0, 3.0]]),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (name, Sigma) in zip(axes, covariances.items()):
    # Draw ellipse: points with Mahalanobis distance = 1 and 2
    theta = jnp.linspace(0, 2*jnp.pi, 200)
    circle = jnp.stack([jnp.cos(theta), jnp.sin(theta)])  # (2, 200)

    # Cholesky: Sigma = L L^T, so points on the Mahalanobis sphere map to circle @ L^T
    L = jnp.linalg.cholesky(Sigma)
    for r, ls in [(1, '-'), (2, '--')]:
        ellipse = (L @ circle * r).T + mu
        ax.plot(ellipse[:, 0], ellipse[:, 1], ls, label=f'r={r}')

    ax.set_title(name); ax.set_aspect('equal')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.legend(); ax.grid(True)

plt.tight_layout(); plt.show()
```

**Code 2. ★★** Sample covariance convergence:

```python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# True covariance
Sigma_true = jnp.array([[3.0, 2.0], [2.0, 2.0]])
mu_true = jnp.array([1.0, 2.0])

# Cholesky decomposition to sample from the Gaussian
L = np.linalg.cholesky(np.array(Sigma_true))

errors = []
sample_sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]

rng = np.random.default_rng(42)
for m in sample_sizes:
    z = rng.normal(size=(m, 2))
    samples = jnp.array(z @ L.T + np.array(mu_true))

    X_c = samples - samples.mean(axis=0)
    Sigma_hat = (X_c.T @ X_c) / (m - 1)

    err = jnp.linalg.norm(Sigma_hat - Sigma_true, ord='fro')
    errors.append(float(err))
    print(f"m={m:5d}: ||Σ̂ - Σ||_F = {err:.4f}")

plt.loglog(sample_sizes, errors, 'o-')
plt.xlabel("Sample size m"); plt.ylabel("Frobenius error")
plt.title("Covariance estimation error vs sample size")
plt.grid(True); plt.show()
```

**Code 3. ★★★** Fit a GMM manually (EM algorithm):

```python
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Generate mixture data
rng = np.random.default_rng(0)
n1, n2 = 100, 100
X1 = rng.multivariate_normal([0, 0],  [[1, 0.5], [0.5, 1]], n1)
X2 = rng.multivariate_normal([5, 3],  [[2, -0.3], [-0.3, 1]], n2)
X = jnp.array(np.vstack([X1, X2]))

# Implement EM for K=2 Gaussians
K, m, d = 2, len(X), 2

def gaussian_pdf(x, mu, Sigma):
    diff = x - mu
    exponent = -0.5 * diff @ jnp.linalg.inv(Sigma) @ diff
    normaliser = jnp.sqrt((2*jnp.pi)**d * jnp.linalg.det(Sigma))
    return jnp.exp(exponent) / normaliser

# Initialise
mus    = jnp.array([[0.0, 0.0], [4.0, 3.0]])
Sigmas = jnp.array([jnp.eye(2), jnp.eye(2)])
pis    = jnp.array([0.5, 0.5])

for iteration in range(30):
    # E-step: compute responsibilities
    r = jnp.zeros((m, K))
    for k in range(K):
        for i in range(m):
            r = r.at[i, k].set(float(pis[k]) * float(gaussian_pdf(X[i], mus[k], Sigmas[k])))
    r = r / r.sum(axis=1, keepdims=True)

    # M-step: update parameters
    Nk = r.sum(axis=0)
    mus    = (r.T @ X) / Nk[:, None]
    pis    = Nk / m
    Sigmas_new = []
    for k in range(K):
        diff = X - mus[k]
        Sk = (r[:, k:k+1] * diff).T @ diff / Nk[k]
        Sigmas_new.append(Sk)
    Sigmas = jnp.array(Sigmas_new)

    if iteration % 10 == 0:
        print(f"Iter {iteration}: mu1={mus[0]}, mu2={mus[1]}")

print(f"\nFinal means:\n  Component 1: {mus[0]}\n  Component 2: {mus[1]}")
print(f"True means:   [0,0] and [5,3]")
```

---

## Solutions

<details>
<summary><strong>P1 Solution</strong></summary>

(a) $\boldsymbol{\mu} = [3, 4]^T$

(b) $\tilde{X} = \begin{bmatrix}-2 & 0\\0 & -2\\2 & 2\end{bmatrix}$

(c) $\tilde{X}^T\tilde{X} = \begin{bmatrix}4+0+4 & 0+0+4\\0+0+4 & 0+4+4\end{bmatrix} = \begin{bmatrix}8 & 4\\4 & 8\end{bmatrix}$, so $\Sigma = \frac{1}{2}\begin{bmatrix}8&4\\4&8\end{bmatrix} = \begin{bmatrix}4&2\\2&4\end{bmatrix}$

(d) $\text{Var}(x_1) = 4$, $\text{Var}(x_2) = 4$, $\text{Cov}(x_1, x_2) = 2 > 0$ — features are **positively correlated**.

</details>

<details>
<summary><strong>P2 Solutions</strong></summary>

(a) Eigenvalues $2, 3 > 0$ → **PD**  
(b) Eigenvalues: $\det(A-\lambda I) = (1-\lambda)^2 - 4 = 0 \Rightarrow \lambda = 3, -1$. One negative eigenvalue → **neither** (indefinite)  
(c) $\det = 4-4 = 0$, eigenvalues $0, 5$ → **PSD** (not PD)  
(d) Eigenvalues: $(3-\lambda)^2 - 1 = 0 \Rightarrow \lambda = 2, 4 > 0$ → **PD**

</details>

<details>
<summary><strong>P4 Solution</strong></summary>

$\Sigma^{-1} = \begin{bmatrix}1/9 & 0\\0 & 1\end{bmatrix}$

(a) Euclidean: $d(\mathbf{p}_1) = 3$, $d(\mathbf{p}_2) = 1$, $d(\mathbf{p}_3) = \sqrt{2} \approx 1.41$

(b) Mahalanobis: $d_M(\mathbf{p}_1) = \sqrt{9/9 + 0} = 1$, $d_M(\mathbf{p}_2) = \sqrt{0 + 1} = 1$, $d_M(\mathbf{p}_3) = \sqrt{1/9 + 1} \approx 1.05$

(c) $\mathbf{p}_1$ is at $(3,0)$ — exactly 1 standard deviation away in the $x_1$ direction (since $\sigma_1 = 3$). $\mathbf{p}_2$ is at $(0,1)$ — exactly 1 standard deviation in the $x_2$ direction (since $\sigma_2 = 1$). Both are equally "surprising." Euclidean said $\mathbf{p}_1$ was farther only because it ignores the fact that $x_1$ is naturally more variable.

</details>
