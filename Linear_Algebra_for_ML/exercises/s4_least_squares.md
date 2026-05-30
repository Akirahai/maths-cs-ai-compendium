# Exercises — Section 4: Solving Systems — Least Squares

> **Difficulty key**: ★ warm-up · ★★ core · ★★★ applied ML · ★★★★ challenge

---

## Conceptual Questions

**C1. ★** When does $A\mathbf{x} = \mathbf{b}$ have (a) exactly one solution, (b) no solution, (c) infinitely many solutions? Connect each case to the rank of $A$.

**C2. ★★** In linear regression, why do we minimise $\|X\mathbf{w} - \mathbf{y}\|^2$ (squared residuals) rather than $\|X\mathbf{w} - \mathbf{y}\|$ (absolute residuals)? Give one algebraic reason and one geometric reason.

**C3. ★★** The normal equations $(X^TX)\mathbf{w} = X^T\mathbf{y}$ require inverting $X^TX$. When is $X^TX$ singular (uninvertible)? Give two concrete examples from a real dataset where this happens.

**C4. ★★** Ridge regression adds $\lambda I$ to $X^TX$ before inverting: $\mathbf{w}^* = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$. Why does this always produce an invertible matrix (for $\lambda > 0$)? What is the cost of this fix?

---

## Pencil-and-Paper Problems

**P1. ★** Solve each system exactly using Gaussian elimination or the inverse formula:

(a) $\begin{cases}2x + y = 5\\x + 3y = 10\end{cases}$

(b) $\begin{cases}x - 2y + z = 0\\2x + y - 3z = 5\\4x - 3y - z = 5\end{cases}$

(c) $\begin{cases}x + 2y = 4\\2x + 4y = 8\end{cases}$ — describe the solution set geometrically

**P2. ★★** Derive the normal equations from scratch. Starting from the loss:

$$\mathcal{L}(\mathbf{w}) = \|X\mathbf{w} - \mathbf{y}\|^2 = (X\mathbf{w} - \mathbf{y})^T(X\mathbf{w} - \mathbf{y})$$

(a) Expand the expression.  
(b) Take $\nabla_\mathbf{w}\mathcal{L} = 0$ (use the identities $\nabla_\mathbf{w}(\mathbf{w}^TA\mathbf{w}) = 2A\mathbf{w}$ for symmetric $A$, and $\nabla_\mathbf{w}(\mathbf{b}^T\mathbf{w}) = \mathbf{b}$).  
(c) Obtain $\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$.

**P3. ★★** Fit a line $y = w_0 + w_1 x$ to the following data using the normal equations:

| $x$ | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| $y$ | 2 | 3 | 5 | 4 | 6 |

(a) Build the design matrix $X$ (with a column of 1s for the intercept).  
(b) Compute $X^TX$ and $X^T\mathbf{y}$.  
(c) Solve the normal equations to find $w_0$ and $w_1$.  
(d) Compute the residuals $\mathbf{e} = \mathbf{y} - X\mathbf{w}^*$ and the MSE.  
(e) Verify that the residuals are orthogonal to the columns of $X$: $X^T\mathbf{e} = \mathbf{0}$.

**P4. ★★** The pseudo-inverse satisfies $A^+ = (A^TA)^{-1}A^T$ when $A$ has full column rank.

For $A = \begin{bmatrix}1 & 0\\0 & 1\\1 & 1\end{bmatrix}$:

(a) Compute $A^TA$ and $(A^TA)^{-1}$.  
(b) Compute $A^+ = (A^TA)^{-1}A^T$.  
(c) Verify that $A^+A = I$ (the left inverse property).  
(d) Does $AA^+ = I$? If not, what is $AA^+$?

**P5. ★★★** (Underdetermined system). Suppose $A$ is $2 \times 4$ with rank 2 — more unknowns than equations. There are infinitely many solutions. The **minimum-norm solution** is $\mathbf{x}^* = A^T(AA^T)^{-1}\mathbf{b}$.

For $A = \begin{bmatrix}1 & 0 & 2 & 1\\0 & 1 & 1 & 2\end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}3\\4\end{bmatrix}$:

(a) Verify $A$ has rank 2.  
(b) Compute $\mathbf{x}^*$.  
(c) Verify $A\mathbf{x}^* = \mathbf{b}$.  
(d) Find one other solution $\mathbf{x}'$ by guessing (pick any $x_3, x_4$ and solve for $x_1, x_2$). Verify $\|\mathbf{x}^*\| \leq \|\mathbf{x}'\|$.

---

## Applied ML Problems

**A1. ★★** **Multicollinearity**. A salary dataset has features: [years\_experience, years\_experience\_squared, has\_degree].

(a) Are the first two features linearly independent? What is the rank of the feature matrix?  
(b) What happens when you try to solve the normal equations?  
(c) Two fixes: (i) drop one feature, (ii) ridge regression. Describe the trade-off.

**A2. ★★** **Polynomial regression**. Fit a degree-2 polynomial $y = w_0 + w_1 x + w_2 x^2$ to:

| $x$ | $-2$ | $-1$ | $0$ | $1$ | $2$ |
|-----|------|------|-----|-----|-----|
| $y$ | $4$ | $1$ | $0$ | $1$ | $4$ |

(a) Build the design matrix $X$ with columns $[1, x, x^2]$.  
(b) Solve the normal equations. What is the true underlying function?  
(c) A degree-4 polynomial would have 5 parameters and 5 data points — it fits perfectly ($\mathbf{e} = \mathbf{0}$). Is this better? Why or why not?

**A3. ★★★** **Ridge regression and regularisation path**. Given the ill-conditioned dataset:

$$X = \begin{bmatrix}1 & 1.001\\1 & 0.999\\1 & 1.000\end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix}2\\2\\2\end{bmatrix}$$

(a) Compute $\text{cond}(X^TX)$ (the condition number). Is the system well-conditioned?  
(b) Compute the OLS solution $(X^TX)^{-1}X^T\mathbf{y}$. Is the answer stable?  
(c) For $\lambda \in \{0.001, 0.01, 0.1, 1.0\}$, compute the ridge solution $(X^TX + \lambda I)^{-1}X^T\mathbf{y}$. Plot $\|\mathbf{w}^*_\lambda\|$ vs $\lambda$ — this is the **regularisation path**.  
(d) As $\lambda \to \infty$, what does the ridge solution approach?

**A4. ★★★** **Weighted least squares**. In standard regression, each data point contributes equally to the loss. In **weighted least squares**, points with lower noise get higher weight:

$$\min_\mathbf{w} \sum_i \omega_i (y_i - \mathbf{w}^T\mathbf{x}_i)^2 = \min_\mathbf{w} (X\mathbf{w} - \mathbf{y})^T W (X\mathbf{w} - \mathbf{y})$$

where $W = \text{diag}(\omega_1, \ldots, \omega_m)$.

(a) Take the gradient and derive the weighted normal equations: $\mathbf{w}^* = (X^TWX)^{-1}X^TW\mathbf{y}$.  
(b) For the data in P3, assign weights $[1, 1, 2, 1, 1]$ (trusting the middle point twice as much). Solve and compare to the unweighted solution.  
(c) In a GMM, the E-step assigns soft responsibilities $r_{ik}$ to each point. Show that the M-step update for the mean of component $k$ is a weighted least squares problem.

---

## Coding Exercises

**Code 1. ★★** Compare three ways to solve a least-squares problem:

```python
import jax.numpy as jnp
import time

# Generate a moderately sized regression problem
import numpy as np
rng = np.random.default_rng(0)
n_samples, n_features = 500, 10
X = jnp.array(np.column_stack([np.ones(n_samples),
                                 rng.normal(size=(n_samples, n_features - 1))]))
true_w = jnp.array(rng.normal(size=n_features))
y = X @ true_w + jnp.array(rng.normal(size=n_samples) * 0.1)

# Method 1: Normal equations via explicit inverse (NEVER do this in production)
w1 = jnp.linalg.inv(X.T @ X) @ X.T @ y

# Method 2: jnp.linalg.solve (better — uses LU decomposition internally)
w2 = jnp.linalg.solve(X.T @ X, X.T @ y)

# Method 3: jnp.linalg.lstsq (best — uses SVD internally, most numerically stable)
w3, _, _, _ = jnp.linalg.lstsq(X, y, rcond=None)

print(f"Max difference method 1 vs 3: {jnp.max(jnp.abs(w1 - w3)):.2e}")
print(f"Max difference method 2 vs 3: {jnp.max(jnp.abs(w2 - w3)):.2e}")
print(f"True weights:      {true_w[:5]}")
print(f"Recovered weights: {w3[:5]}")
```

**Code 2. ★★** Demonstrate the multicollinearity problem and ridge fix:

```python
import jax.numpy as jnp
import numpy as np

rng = np.random.default_rng(1)
n = 50
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(size=n) * 0.001   # nearly identical to x1
y  = 2 * x1 + 3 + rng.normal(size=n) * 0.1

X = jnp.column_stack([jnp.ones(n), x1, x2])  # bias + two nearly-identical columns
y = jnp.array(y)

# 1. Check the condition number of X^T X
XtX = X.T @ X
cond = jnp.linalg.cond(XtX)
print(f"Condition number of X^T X: {cond:.2e}")  # will be enormous

# 2. OLS solution — observe instability
w_ols = jnp.linalg.lstsq(X, y, rcond=None)[0]
print(f"OLS weights: {w_ols}")   # one of w1, w2 will be huge

# 3. Ridge solutions for various lambda
for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
    w_ridge = jnp.linalg.solve(XtX + lam * jnp.eye(3), X.T @ y)
    print(f"λ={lam:.2f} → w={w_ridge}  ||w||={jnp.linalg.norm(w_ridge):.2f}")
```

**Code 3. ★★★** Implement polynomial regression and observe overfitting:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
x_train = jnp.linspace(0, 1, 10)
y_train = jnp.sin(2 * jnp.pi * x_train) + jnp.array(rng.normal(size=10) * 0.2)

x_test = jnp.linspace(0, 1, 100)

def poly_design_matrix(x, degree):
    """Return Vandermonde matrix with columns [1, x, x^2, ..., x^degree]."""
    return jnp.column_stack([x ** d for d in range(degree + 1)])

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, degree in enumerate([1, 3, 9, 15]):
    X_tr = poly_design_matrix(x_train, degree)
    X_te = poly_design_matrix(x_test, degree)

    # Ridge to stabilise high-degree fits
    lam = 1e-6 if degree < 9 else 1e-3
    w = jnp.linalg.solve(X_tr.T @ X_tr + lam * jnp.eye(degree + 1), X_tr.T @ y_train)

    y_pred = X_te @ w
    train_mse = jnp.mean((y_train - X_tr @ w) ** 2)

    axes[idx].scatter(x_train, y_train, color='black', s=30, label='data')
    axes[idx].plot(x_test, y_pred, color='blue', label=f'deg={degree}')
    axes[idx].plot(x_test, jnp.sin(2 * jnp.pi * x_test), 'r--', label='true')
    axes[idx].set_title(f"Degree {degree}\nMSE={train_mse:.4f}")
    axes[idx].legend(fontsize=7); axes[idx].set_ylim(-2, 2)

plt.tight_layout(); plt.show()
```

---

## Solutions

<details>
<summary><strong>P1(a) Solution</strong></summary>

From row 1: $y = 5 - 2x$. Substituting into row 2: $x + 3(5-2x) = 10 \Rightarrow -5x = -5 \Rightarrow x = 1$, $y = 3$.

Solution: $(x, y) = (1, 3)$.

</details>

<details>
<summary><strong>P3 Solution (key steps)</strong></summary>

$X = \begin{bmatrix}1&1\\1&2\\1&3\\1&4\\1&5\end{bmatrix}$

$X^TX = \begin{bmatrix}5 & 15\\15 & 55\end{bmatrix}$, $\quad X^T\mathbf{y} = \begin{bmatrix}20\\67\end{bmatrix}$

$\det(X^TX) = 5\cdot55 - 15^2 = 50$

$\mathbf{w}^* = \frac{1}{50}\begin{bmatrix}55 & -15\\-15 & 5\end{bmatrix}\begin{bmatrix}20\\67\end{bmatrix} = \frac{1}{50}\begin{bmatrix}1100-1005\\-300+335\end{bmatrix} = \begin{bmatrix}1.9\\0.7\end{bmatrix}$

Line: $y = 1.9 + 0.7x$.

Residuals: $[2-2.6, 3-3.3, 5-4.0, 4-4.7, 6-5.4] = [-0.6, -0.3, 1.0, -0.7, 0.6]$.

MSE $= (0.36 + 0.09 + 1.00 + 0.49 + 0.36)/5 = 0.46$.

</details>

<details>
<summary><strong>P4(d) Note</strong></summary>

$AA^+$ is NOT the identity in general when $A$ is not square. It is the **projection matrix** $H = X(X^TX)^{-1}X^T$ that projects onto the column space of $A$. So $AA^+\mathbf{y} = \hat{\mathbf{y}}$ (the linear regression prediction).

</details>
