# Exercises — Section 5: Constrained Optimisation — Lagrange Multipliers

> **Difficulty key**: ★ warm-up · ★★ core · ★★★ applied ML · ★★★★ challenge

---

## Conceptual Questions

**C1. ★** Explain in one sentence why you cannot just set $\nabla f = 0$ to solve a constrained optimisation problem. Draw a sketch for the 2D case where the unconstrained minimum lies off the constraint curve.

**C2. ★★** The Lagrangian is $\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})$. The condition $\nabla_\mathbf{x}\mathcal{L} = 0$ says $\nabla f = -\lambda \nabla g$. What does it mean geometrically for $\nabla f$ and $\nabla g$ to be parallel at the constrained optimum?

**C3. ★★** In the KKT conditions for inequality constraints $g(\mathbf{x}) \leq 0$, there is a **complementary slackness** condition: $\lambda g(\mathbf{x}) = 0$. This means either the constraint is active ($g = 0$) or the multiplier is zero ($\lambda = 0$). Give the physical interpretation of each case in the SVM context.

**C4. ★★★** The SVM dual has $m$ variables ($\alpha_1, \ldots, \alpha_m$, one per training point) while the primal has $n+1$ variables ($\mathbf{w}$ and $b$, where $n$ is the feature dimension). When is the dual formulation computationally *cheaper* than the primal? When is it more expensive?

---

## Pencil-and-Paper Problems

**P1. ★** Find the extrema of $f(x, y) = x + y$ subject to $x^2 + y^2 = 1$ (the unit circle). Use Lagrange multipliers and interpret the result geometrically.

**P2. ★** Minimise $f(x, y) = x^2 + y^2$ (distance² from origin) subject to $3x + 4y = 25$.

(a) Form $\mathcal{L}$ and solve the stationarity conditions.  
(b) What is the closest point on the line to the origin? What is the minimum distance?  
(c) What is $\lambda^*$? Recall that $\lambda$ equals $\partial f^* / \partial c$ (the rate of change of the optimal value as the constraint $g = c$ shifts). Verify this interpretation by solving for $c = 26$ and checking the change in $f^*$.

**P3. ★★** Maximise $f(x, y, z) = xyz$ subject to $x + y + z = 1$ with $x, y, z \geq 0$.

(a) Ignore the inequality constraints and apply Lagrange multipliers for the equality only.  
(b) Show the answer is $x^* = y^* = z^* = 1/3$, with $f^* = 1/27$.  
(c) Confirm by the AM-GM inequality: for $x+y+z = 1$, $\frac{x+y+z}{3} \geq (xyz)^{1/3}$, so $xyz \leq (1/3)^3$. This is the "symmetry shortcut" generalised.

**P4. ★★** Minimise $f(x,y) = 2x^2 + y^2$ subject to $x + y = 1$ using Lagrange multipliers. Verify your answer by substituting the constraint directly and solving the unconstrained 1D problem.

**P5. ★★** Two equality constraints. Minimise $f(x,y,z) = x^2 + y^2 + z^2$ subject to:

$$g_1: x + y + z = 1 \qquad g_2: x + 2y = 0$$

Form $\mathcal{L} = f + \lambda_1 g_1 + \lambda_2 g_2$, derive the stationarity system, and solve.

**P6. ★★★** **Maximum entropy distribution**. Maximise the Shannon entropy:

$$H = -\sum_{i=1}^{n} p_i \ln p_i$$

subject to:
- $\sum_{i=1}^{n} p_i = 1$ (probabilities sum to 1)
- $p_i \geq 0$ for all $i$

(a) Form the Lagrangian $\mathcal{L} = -\sum p_i \ln p_i + \lambda(\sum p_i - 1)$.  
(b) Take $\partial \mathcal{L}/\partial p_i = 0$ for each $i$. Show $-\ln p_i - 1 + \lambda = 0$.  
(c) Solve for $p_i$ and apply the constraint to find $\lambda$.  
(d) Show the answer is the **uniform distribution** $p_i = 1/n$, $H_{\max} = \ln n$.  
(e) **ML connection**: Naive Bayes assumes feature independence (maximum entropy given marginals). Logistic regression is a maximum-entropy classifier given feature expectations.

---

## Applied ML Problems

**A1. ★★** **Hard-margin SVM — 1D example**. Given points in 1D:
- Class $+1$: $x = 3, 4$
- Class $-1$: $x = 0, 1$

The SVM decision boundary is a single point $b$ (threshold), and the margin is the gap between classes divided by the "weight" $w$.

(a) Write the SVM primal: minimise $\frac{1}{2}w^2$ subject to $y_i(wx_i + b) \geq 1$.  
(b) Form the Lagrangian with multipliers $\alpha_1, \alpha_2, \alpha_3, \alpha_4$.  
(c) Derive the KKT conditions. By symmetry, the boundary is at $x = 2$. Find $w$, $b$, and the nonzero $\alpha_i$.  
(d) Which points are support vectors? What is the margin?

**A2. ★★** **Dual derivation step-by-step**. For the general hard-margin SVM:

Primal: $\min \frac{1}{2}\|\mathbf{w}\|^2$ s.t. $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \geq 1$.

Lagrangian: $\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_i \alpha_i[y_i(\mathbf{w}\cdot\mathbf{x}_i + b) - 1]$.

(a) Compute $\nabla_\mathbf{w}\mathcal{L} = 0$ and show $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$.  
(b) Compute $\partial\mathcal{L}/\partial b = 0$ and show $\sum_i \alpha_i y_i = 0$.  
(c) Substitute (a) into $\mathcal{L}$ to eliminate $\mathbf{w}$. Show you get:

$$\mathcal{L}_D(\boldsymbol{\alpha}) = \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_iy_j(\mathbf{x}_i\cdot\mathbf{x}_j)$$

(d) Maximise $\mathcal{L}_D$ subject to $\alpha_i \geq 0$ and $\sum_i \alpha_i y_i = 0$ — this is the dual.

**A3. ★★★** **Soft-margin SVM and KKT**. The soft-margin SVM introduces slack variables $\xi_i \geq 0$:

$$\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i\xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

with multipliers $\alpha_i \geq 0$ for the margin constraints and $\mu_i \geq 0$ for the non-negativity of $\xi_i$.

(a) Write the full Lagrangian.  
(b) Derive the stationarity conditions: $\nabla_\mathbf{w}\mathcal{L} = 0$, $\partial\mathcal{L}/\partial b = 0$, $\partial\mathcal{L}/\partial\xi_i = 0$. Show the last gives $\alpha_i + \mu_i = C$.  
(c) Write out the KKT complementary slackness conditions. For each of the three cases ($\alpha_i = 0$; $0 < \alpha_i < C$; $\alpha_i = C$), describe the type of training point (non-support vector, support vector on margin, support vector inside margin or misclassified).

**A4. ★★★★** **Regularisation as a Lagrangian**. Ridge regression minimises:

$$\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$$

But equivalently, this solves the **constrained problem**:

$$\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|^2 \quad \text{s.t.} \quad \|\mathbf{w}\|^2 \leq t$$

for some value of $t$ that depends on $\lambda$.

(a) Form the Lagrangian for the constrained version.  
(b) Show the stationarity condition gives $(X^TX + \lambda I)\mathbf{w} = X^T\mathbf{y}$ — exactly the ridge formula.  
(c) Lasso regression uses $\|\mathbf{w}\|_1 \leq t$ instead. Why does this produce **sparse** solutions (many $w_i = 0$) while ridge does not? (Hint: sketch the L1 ball and elliptical contours of the loss in 2D.)

---

## Coding Exercises

**Code 1. ★** Numerically verify P1 using scipy:

```python
from scipy.optimize import minimize
import numpy as np

# Minimise -f = -(x+y) subject to x^2 + y^2 - 1 = 0
result = minimize(
    fun=lambda xy: -(xy[0] + xy[1]),
    x0=[0.5, 0.5],
    constraints={'type': 'eq', 'fun': lambda xy: xy[0]**2 + xy[1]**2 - 1},
    method='SLSQP'
)
print(f"Optimal (x,y) = {result.x}")  # should be [1/sqrt(2), 1/sqrt(2)]
print(f"Maximum f*   = {-result.fun:.6f}")  # should be sqrt(2)
```

**Code 2. ★★** Solve the maximum-entropy problem numerically and verify the uniform distribution:

```python
from scipy.optimize import minimize
import numpy as np

n = 6  # number of outcomes

def neg_entropy(p):
    p = np.clip(p, 1e-12, 1)   # avoid log(0)
    return np.sum(p * np.log(p))

constraints = [
    {'type': 'eq',  'fun': lambda p: np.sum(p) - 1},
]
bounds = [(0, 1)] * n

result = minimize(neg_entropy, x0=np.ones(n)/n,
                  constraints=constraints, bounds=bounds, method='SLSQP')

print(f"Optimal p: {result.x}")                          # should be [1/6, ..., 1/6]
print(f"Max entropy: {-result.fun:.6f}")                 # should be ln(6) ≈ 1.7918
print(f"ln(n) = {np.log(n):.6f}")
```

**Code 3. ★★★** Train a hard-margin SVM using the dual and visualise support vectors:

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Linearly separable 2D data
X = np.array([[1, 2], [2, 3], [3, 3],   # class +1
              [-1, -1], [-2, -2], [-1, -3]])  # class -1
y = np.array([1, 1, 1, -1, -1, -1], dtype=float)
m = len(y)

# Gram matrix weighted by labels: Q_ij = y_i y_j (x_i . x_j)
K = X @ X.T
Q = np.outer(y, y) * K

# Dual: maximise sum(alpha) - 0.5 * alpha^T Q alpha
# = minimise -sum(alpha) + 0.5 * alpha^T Q alpha
def dual_objective(alpha):
    return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

def dual_gradient(alpha):
    return Q @ alpha - np.ones(m)

constraints = [{'type': 'eq', 'fun': lambda a: np.dot(a, y)}]
bounds = [(0, None)] * m

result = minimize(dual_objective, x0=np.ones(m) * 0.1,
                  jac=dual_gradient, bounds=bounds,
                  constraints=constraints, method='SLSQP')

alphas = result.x
print(f"Alphas: {alphas.round(4)}")

# Recover w and b
w = (alphas * y) @ X
sv_mask = alphas > 1e-4
b = np.mean(y[sv_mask] - X[sv_mask] @ w)

print(f"w = {w}, b = {b:.4f}")
print(f"Margin = {2 / np.linalg.norm(w):.4f}")
print(f"Support vectors: {X[sv_mask]}")

# Plot
xx = np.linspace(-4, 4, 100)
yy = -(w[0] * xx + b) / w[1]
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=60, label='+1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], c='red', s=60, label='-1')
plt.scatter(X[sv_mask, 0], X[sv_mask, 1], s=200, facecolors='none',
            edgecolors='black', linewidths=2, label='support vectors')
plt.plot(xx, yy, 'k-', label='decision boundary')
plt.plot(xx, yy + 1/np.linalg.norm(w), 'k--')
plt.plot(xx, yy - 1/np.linalg.norm(w), 'k--')
plt.legend(); plt.grid(True); plt.show()
```

---

## Solutions

<details>
<summary><strong>P1 Solution</strong></summary>

$\mathcal{L} = x + y + \lambda(x^2 + y^2 - 1)$

$\partial\mathcal{L}/\partial x = 1 + 2\lambda x = 0 \Rightarrow x = -1/(2\lambda)$

$\partial\mathcal{L}/\partial y = 1 + 2\lambda y = 0 \Rightarrow y = -1/(2\lambda)$

So $x = y$. Substituting into constraint: $2x^2 = 1 \Rightarrow x = \pm 1/\sqrt{2}$.

- Maximum: $(1/\sqrt{2}, 1/\sqrt{2})$, $f^* = \sqrt{2}$
- Minimum: $(-1/\sqrt{2}, -1/\sqrt{2})$, $f^* = -\sqrt{2}$

Geometrically: we're finding where the level sets of $f = x+y$ (diagonal lines) are tangent to the unit circle.

</details>

<details>
<summary><strong>P2 Solution</strong></summary>

$\mathcal{L} = x^2 + y^2 + \lambda(3x + 4y - 25)$

$\partial\mathcal{L}/\partial x = 2x + 3\lambda = 0 \Rightarrow x = -3\lambda/2$

$\partial\mathcal{L}/\partial y = 2y + 4\lambda = 0 \Rightarrow y = -2\lambda$

Constraint: $3(-3\lambda/2) + 4(-2\lambda) = 25 \Rightarrow -9\lambda/2 - 8\lambda = 25 \Rightarrow \lambda = -2$

So $x = 3$, $y = 4$. Distance $= \sqrt{9+16} = 5$. $f^* = 25$.

The multiplier $\lambda = -2$ means: if the RHS of the constraint increases by 1 (from 25 to 26), the minimum distance² increases by 2 (from 25 to 27), so the minimum distance increases from 5 to $\sqrt{27} \approx 5.196$.

</details>

<details>
<summary><strong>P6 Solution (key steps)</strong></summary>

Setting $\partial\mathcal{L}/\partial p_i = 0$: $-(\ln p_i + 1) + \lambda = 0 \Rightarrow p_i = e^{\lambda - 1}$.

Since all $p_i$ are equal to the same constant $c = e^{\lambda-1}$, and $\sum p_i = 1$: $nc = 1 \Rightarrow c = 1/n$.

So $p_i = 1/n$ for all $i$ — the uniform distribution. Maximum entropy: $H = -n \cdot \frac{1}{n}\ln\frac{1}{n} = \ln n$.

</details>
