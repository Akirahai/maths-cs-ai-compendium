# Linear Algebra for Machine Learning — 1-Hour Tutorial

> **Audience**: Students beginning an ML course who have elementary maths background.  
> **Goal**: Build the exact linear algebra intuition and mechanics needed for every topic in the ML syllabus — from Perceptrons to PCA — in one sitting.  
> **Format**: Read top to bottom. Run every code block. Each section ends with a direct callout to where it appears in your course.

---

## Roadmap

| Section | Time | Covers in Course |
|---------|------|-----------------|
| 1. Vectors as Data | 10 min | All weeks — feature representation |
| 2. Similarity and Distance | 10 min | Wk 1–4: Perceptron, SVM, K-Means |
| 3. Matrices and Transformations | 12 min | Wk 2, 5: Linear Regression, Neural Nets |
| 4. Solving Systems — Least Squares | 8 min | Wk 2: Linear Regression |
| 5. Constrained Optimisation: Lagrange Multipliers | 12 min | Wk 4: SVM margin maximisation |
| 6. Covariance and Positive Definiteness | 8 min | Wk 6: GMM, Naive Bayes |
| 7. Eigenvalues, SVD, and PCA | 10 min | Wk 9: Dimensionality Reduction |
| 8. Quick Reference for the Course | 2 min | All weeks |

---

## Section 1 — Vectors as Data (10 min)

### What is a vector?

A **vector** is an ordered list of numbers. In ML, it is always a data point.

$$\mathbf{x} = [x_1, x_2, \ldots, x_n]$$

**Example**: Represent a student as a vector of features:

$$\mathbf{x} = [\underbrace{22}_{\text{age}},\; \underbrace{170}_{\text{height (cm)}},\; \underbrace{65}_{\text{weight (kg)}},\; \underbrace{3.8}_{\text{GPA}}]$$

This is a vector in $\mathbb{R}^4$ — four-dimensional space. Every ML algorithm treats your data this way: each sample is a vector, and the whole dataset is a collection of them.

### Vector space rules (why they matter)

Two rules govern everything:

1. **Vector addition**: $\mathbf{a} + \mathbf{b} = (a_1 + b_1,\; a_2 + b_2,\; \ldots)$ — combine two data points component-wise.
2. **Scalar multiplication**: $c\mathbf{a} = (ca_1,\; ca_2,\; \ldots)$ — stretch or shrink a data point.

These two operations, closed within the space, are what makes it a **vector space**. Every ML model lives in one.

### Magnitude and unit vectors

The **magnitude** (length) of a vector is the straight-line distance from the origin to its tip:

$$\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

A **unit vector** has magnitude exactly 1. You get one by normalising:

$$\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}$$

Normalisation strips away "how big" and keeps only "which direction." You will use it constantly (e.g., normalising inputs before training, computing cosine similarity).

### Linear independence — why it matters for ML

A set of vectors is **linearly independent** if no vector in the set can be built from the others. In ML this means: each feature adds genuinely new information. When features are perfectly correlated (linearly dependent), they are redundant — this is why PCA (Week 9) removes them.

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Three students as feature vectors: [age, height_cm, weight_kg]
alice = jnp.array([22.0, 165.0, 58.0])
bob   = jnp.array([25.0, 180.0, 82.0])
carol = jnp.array([21.0, 162.0, 55.0])

# Magnitude of Alice's vector
print(f"||alice|| = {jnp.linalg.norm(alice):.2f}")

# Normalise
alice_hat = alice / jnp.linalg.norm(alice)
print(f"Unit vector: {alice_hat}")
print(f"||alice_hat|| = {jnp.linalg.norm(alice_hat):.4f}")  # should be 1.0

# Vector addition and scalar multiplication
mean_student = (alice + bob + carol) / 3
print(f"Mean student: {mean_student}")
```

> **Course connection — all weeks**: Every dataset you feed into a classifier, regressor, or clustering algorithm is a matrix of row vectors. Knowing what a vector space is tells you *why* operations like averaging, projecting, and transforming are legal.

### Quick Exercises — Section 1

**Q1.** A house is described by the vector $\mathbf{x} = [120, 3, 15, 2]$ (area m², bedrooms, age, bathrooms). What is its magnitude $\|\mathbf{x}\|$? Compute the unit vector $\hat{\mathbf{x}}$ and verify its magnitude is 1.

**Q2.** Are the vectors $\mathbf{a} = [2, 4, 6]$ and $\mathbf{b} = [1, 2, 3]$ linearly independent? If a dataset has two features with this relationship, what should you do before training?

**Q3.** A fruit's feature vector is $\mathbf{f} = [150, 8, 3.5]$ (weight g, sweetness/10, price/kg). A second fruit is $\mathbf{g} = [80, 6, 5.0]$. Compute the mean fruit vector and interpret each component.

**Q4. (Code)** Create a $5 \times 3$ data matrix for 5 emails, where each row is a bag-of-words count vector over vocabulary ["spam", "offer", "meeting"]. Mean-centre the matrix (subtract the column means). What does the centred matrix represent?

> See `exercises/s1_vectors_as_data.md` for full worked solutions and additional problems.

---

## Section 2 — Similarity and Distance (10 min)

### Norms: measuring size

There are multiple ways to measure the "size" of a vector. The most common:

| Name | Formula | When used |
|------|---------|-----------|
| L2 (Euclidean) | $\|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}$ | Default — KNN, K-Means, SVM |
| L1 (Manhattan) | $\|\mathbf{v}\|_1 = \sum |v_i|$ | Sparse data, robust regression |
| L$\infty$ (Max) | $\|\mathbf{v}\|_\infty = \max_i |v_i|$ | Worst-case analysis |

### Distance: measuring how far apart two points are

A distance between two vectors $\mathbf{u}$ and $\mathbf{v}$ is just the norm of their difference:

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|$$

Using L2 gives **Euclidean distance** — the straight line between two points. Using L1 gives **Manhattan distance** — city-block travel. **K-Means (Week 3)** assigns each point to the closest centroid by Euclidean distance. **K-Medoids** does the same but is more robust because it uses actual data points as centres, and the choice of norm matters: L1 is less affected by outliers than L2.

### The dot product and what it measures

The **dot product** of two vectors is a scalar:

$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

Its geometric meaning is everything:

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta$$

where $\theta$ is the angle between them. So the dot product answers: **"how much do these two vectors agree in direction?"**

- $\mathbf{a} \cdot \mathbf{b} > 0$: vectors point roughly the same way (angle $< 90°$)
- $\mathbf{a} \cdot \mathbf{b} = 0$: vectors are perpendicular — completely independent
- $\mathbf{a} \cdot \mathbf{b} < 0$: vectors point against each other (angle $> 90°$)

### Cosine similarity

To compare direction without caring about magnitude, normalise both vectors first:

$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}$$

Result ranges from $-1$ (opposite) to $+1$ (identical direction).

### The Perceptron decision rule (Week 1)

The Perceptron classifies by computing a dot product between the weight vector $\mathbf{w}$ and the input $\mathbf{x}$, then checking the sign:

$$\hat{y} = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$$

The weight vector $\mathbf{w}$ defines a **hyperplane** (a flat decision boundary). Points on one side of the hyperplane satisfy $\mathbf{w} \cdot \mathbf{x} + b > 0$, points on the other side satisfy $< 0$. Everything the Perceptron does is a dot product.

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, -1.0, 2.0])

# Dot product
dot = jnp.dot(a, b)
angle_deg = jnp.degrees(jnp.arccos(dot / (jnp.linalg.norm(a) * jnp.linalg.norm(b))))
print(f"Dot product: {dot}")
print(f"Angle: {angle_deg:.1f}°")

# Cosine similarity
cos_sim = dot / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
print(f"Cosine similarity: {cos_sim:.4f}")

# Euclidean vs Manhattan distance
u = jnp.array([1.0, 2.0])
v = jnp.array([4.0, 6.0])
euclidean = jnp.linalg.norm(u - v)
manhattan = jnp.sum(jnp.abs(u - v))
print(f"Euclidean: {euclidean:.2f},  Manhattan: {manhattan:.2f}")

# Perceptron decision on a simple example
w = jnp.array([0.5, -1.0])      # learned weights
x = jnp.array([2.0, 1.0])       # input sample
b_bias = 0.1
score = jnp.dot(w, x) + b_bias
print(f"Perceptron score: {score:.2f},  Prediction: {'+1' if score > 0 else '-1'}")
```

> **Course connection**:
> - **Week 1–2 (Perceptron, Hinge Loss)**: the score $\mathbf{w} \cdot \mathbf{x}$ is computed every forward pass.
> - **Week 3 (K-Means, K-Medoids)**: cluster assignment = find the centroid with minimum $\|\mathbf{x} - \boldsymbol{\mu}_k\|$.
> - **Week 4 (SVMs)**: the margin is $\frac{2}{\|\mathbf{w}\|}$; maximising it means minimising the weight norm.

### Quick Exercises — Section 2

**Q1.** A Perceptron has weights $\mathbf{w} = [3, -2, 1]$ and bias $b = -1$. Classify these two inputs and explain whether each is on the positive or negative side of the hyperplane:
- $\mathbf{x}_1 = [1, 1, 1]$
- $\mathbf{x}_2 = [0, 2, 0]$

**Q2.** Three documents are represented as term-frequency vectors over vocabulary ["cat", "dog", "fish"]:
- $\mathbf{d}_1 = [3, 0, 1]$, $\mathbf{d}_2 = [0, 2, 2]$, $\mathbf{d}_3 = [6, 0, 2]$

Compute the cosine similarity between each pair. Which two documents are most similar? Would Euclidean distance give the same ranking? Why or why not?

**Q3.** Points $\mathbf{p} = (0, 0)$ and $\mathbf{q} = (3, 4)$. Compute L1, L2, and L$\infty$ distances. Now add an outlier point at $(3, 100)$ and recompute. Which norm is most affected? What does this imply for K-Medoids vs K-Means?

**Q4.** In a K-Means iteration, centroids are $\boldsymbol{\mu}_1 = (1, 1)$ and $\boldsymbol{\mu}_2 = (5, 5)$. Assign each point to the nearest centroid: $(2, 2)$, $(4, 3)$, $(0, 3)$, $(6, 4)$. Then recompute the two centroids.

> See `exercises/s2_similarity_distance.md` for full worked solutions and additional problems.

---

## Section 3 — Matrices and Transformations (12 min)

### What is a matrix?

A **matrix** is a rectangular table of numbers. Two interpretations you need:

1. **A dataset**: $m$ rows = $m$ samples, $n$ columns = $n$ features. The entire dataset is one $m \times n$ matrix.
2. **A transformation**: multiplying by a matrix moves vectors from one space to another.

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \leftarrow \text{2 samples, 3 features each}$$

### Matrix-vector multiplication: the forward pass

The single most common operation in ML is multiplying a matrix by a vector:

$$A\mathbf{x} = \begin{bmatrix} \mathbf{a}_1^T \cdot \mathbf{x} \\ \mathbf{a}_2^T \cdot \mathbf{x} \\ \vdots \end{bmatrix}$$

Each output element is a dot product of one row of $A$ with $\mathbf{x}$. A **neural network layer** is exactly this: $\mathbf{y} = W\mathbf{x} + \mathbf{b}$, where $W$ is the weight matrix, $\mathbf{x}$ is the input vector, and $\mathbf{b}$ is the bias.

### Linear transformations

Every matrix represents a **linear transformation** — a function that maps vectors to vectors while preserving addition and scaling. The matrix columns tell you where the basis vectors land:

- **Rotation** — length preserved, direction changed
- **Scaling** — stretches or shrinks along each axis
- **Projection** — collapses one or more dimensions

```math
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
```

This is exactly the kind of transformation a neural network layer applies (minus the activation function).

### Key matrix properties

| Property | Definition | ML relevance |
|----------|-----------|--------------|
| **Transpose** $A^T$ | Swap rows and columns | Appears in every gradient formula |
| **Rank** | Number of linearly independent rows/columns | Low rank → redundant features |
| **Determinant** | Scaling factor of the transformation | 0 → singular, no inverse |
| **Inverse** $A^{-1}$ | Matrix that undoes $A$: $AA^{-1} = I$ | Exists only if $\det(A) \neq 0$ |

### Matrix multiplication: chaining transformations

Multiplying two matrices $C = AB$ chains two transformations: first apply $B$, then apply $A$. Dimensions must be compatible: if $A$ is $m \times k$ and $B$ is $k \times n$, then $C$ is $m \times n$.

**Critical rule**: $AB \neq BA$ in general. Order matters. Think: rotating then translating is different from translating then rotating.

```python
import jax.numpy as jnp

# --- Dataset as a matrix: 4 students, 3 features ---
X = jnp.array([
    [22.0, 165.0, 58.0],  # alice
    [25.0, 180.0, 82.0],  # bob
    [21.0, 162.0, 55.0],  # carol
    [28.0, 175.0, 70.0],  # dave
])  # shape: (4, 3)

print(f"Dataset shape: {X.shape}  ->  {X.shape[0]} samples, {X.shape[1]} features")

# --- Neural network layer: Wx + b ---
W = jnp.array([[0.1, -0.2, 0.3],
               [0.4,  0.0, -0.1]])   # (2, 3): maps 3 features -> 2 outputs
b = jnp.array([0.5, -0.5])           # (2,)

# Forward pass for ONE sample (alice)
alice = X[0]
y = W @ alice + b
print(f"Layer output for alice: {y}")

# Forward pass for ALL samples at once (batch)
Y = X @ W.T + b        # (4, 3) @ (3, 2) + (2,) = (4, 2)
print(f"Batch output shape: {Y.shape}")

# --- Matrix properties ---
A = jnp.array([[2.0, 1.0],
               [1.0, 3.0]])

print(f"\nMatrix A:\n{A}")
print(f"Transpose:\n{A.T}")
print(f"Trace (sum of diagonal): {jnp.trace(A):.2f}")
print(f"Determinant: {jnp.linalg.det(A):.2f}")
print(f"Rank: {jnp.linalg.matrix_rank(A)}")

A_inv = jnp.linalg.inv(A)
print(f"A @ A_inv ≈ I:\n{(A @ A_inv).round(6)}")

# --- Non-commutativity ---
B = jnp.array([[1.0, 2.0], [3.0, 4.0]])
print(f"\nAB == BA? {jnp.allclose(A @ B, B @ A)}")
```

> **Course connection**:
> - **Week 2 (Linear Regression)**: predictions are $\hat{\mathbf{y}} = X\mathbf{w}$ — a matrix times a weight vector.
> - **Week 5 (Neural Networks)**: each layer is $\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$ — matrix multiplication followed by a nonlinearity.
> - **Week 4 (SVMs)**: the kernel trick replaces dot products $\mathbf{x}_i \cdot \mathbf{x}_j$ with kernel evaluations — same linear algebra, higher-dimensional space.

### Quick Exercises — Section 3

**Q1.** Compute by hand:

$$\begin{bmatrix}1 & 2\\3 & 4\\5 & 6\end{bmatrix} \begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}$$

What is the rule at work? Now compute the same matrix times $\begin{bmatrix}2 & 0\\0 & 3\end{bmatrix}$. What transformation does this represent?

**Q2.** A weight matrix $W = \begin{bmatrix}1 & -1\\2 & 0\end{bmatrix}$ maps 2D inputs to 2D outputs. Apply it to $\mathbf{x} = [3, 1]^T$. What is the determinant of $W$? Does $W^{-1}$ exist? If so, compute it and verify $WW^{-1} = I$.

**Q3.** A rotation matrix for $\theta = 45°$ is $R = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & -1\\1 & 1\end{bmatrix}$. Apply it to the unit vector $[1, 0]^T$. Verify that $\|R\mathbf{v}\| = \|\mathbf{v}\|$ (rotations preserve length). Does $RR^T = I$?

**Q4. (Code)** Build a 2-layer neural network forward pass for a batch of 3 samples with 4 input features, a hidden layer of 3 neurons, and 2 output neurons. Use random weights (seed fixed). Verify all intermediate shapes.

> See `exercises/s3_matrices_transformations.md` for full worked solutions and additional problems.

---

## Section 4 — Solving Systems: Least Squares (8 min)

### The system $A\mathbf{x} = \mathbf{b}$

Many ML problems reduce to solving a system of linear equations. Given an $m \times n$ matrix $A$ and a vector $\mathbf{b}$, find $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{b}$.

If $A$ is square and invertible: $\mathbf{x} = A^{-1}\mathbf{b}$.

In practice you use `jnp.linalg.solve(A, b)` — never compute the inverse explicitly (numerically unstable).

### Over-determined systems and least squares

In ML, you have far more equations than unknowns ($m \gg n$): many more samples than features. No exact solution exists. Instead, we find the $\mathbf{w}$ that minimises the **sum of squared residuals**:

$$\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2$$

This is the **Ordinary Least Squares (OLS)** problem. Taking the gradient and setting it to zero gives the **Normal Equations**:

$$X^T X \mathbf{w} = X^T \mathbf{y}$$

$$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

The matrix $(X^T X)^{-1} X^T$ is the **pseudo-inverse** of $X$, written $X^+$. This is the exact closed-form solution for **Linear Regression (Week 2)**.

### Why this matters beyond regression

The pseudo-inverse appears whenever you want the "best possible" linear fit. It also explains why we care about `rank(X)`: if $X$ does not have full column rank (linearly dependent features), $X^T X$ is singular and has no inverse. You get infinitely many equally good solutions. This is the **multicollinearity** problem.

```python
import jax.numpy as jnp

# --- Generate toy data: house size -> price ---
# X: feature matrix with a bias column (intercept trick)
X = jnp.array([
    [1.0, 50.0],    # [bias, size_m2]
    [1.0, 75.0],
    [1.0, 100.0],
    [1.0, 120.0],
    [1.0, 150.0],
])
y = jnp.array([200.0, 280.0, 350.0, 410.0, 500.0])  # price (thousands)

# --- Normal equations: w* = (X^T X)^{-1} X^T y ---
w_star = jnp.linalg.inv(X.T @ X) @ X.T @ y
print(f"Learned weights: intercept={w_star[0]:.2f}, slope={w_star[1]:.2f}")

# Predictions
y_hat = X @ w_star
print(f"Predictions: {y_hat}")
print(f"Actual:      {y}")

# Mean Squared Error
mse = jnp.mean((y - y_hat) ** 2)
print(f"MSE: {mse:.4f}")

# --- numpy's built-in lstsq uses SVD internally (more stable) ---
w_lstsq, _, _, _ = jnp.linalg.lstsq(X, y, rcond=None)
print(f"\nlstsq solution: {w_lstsq}")

# --- Solve a small square system exactly ---
A = jnp.array([[2.0, 1.0],
               [5.0, 3.0]])
b = jnp.array([4.0, 7.0])
x = jnp.linalg.solve(A, b)
print(f"\nExact solution x: {x}")
print(f"Verification A@x: {A @ x}")
```

> **Course connection**:
> - **Week 2 (Linear Regression)**: closed-form solution $\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$ is the Normal Equations derived above.
> - **Week 2 (Gradient Descent for Regression)**: when $(X^TX)^{-1}$ is too expensive to compute (large $n$), we solve iteratively — gradient descent steps towards the same optimum.
> - **Week 4 (SVM dual)**: the SVM dual problem is also a quadratic in a matrix form — same linear algebra intuition.

### Quick Exercises — Section 4

**Q1.** Solve the following system exactly by hand, then verify:

$$\begin{bmatrix}2 & 1\\5 & 3\end{bmatrix}\mathbf{x} = \begin{bmatrix}8\\19\end{bmatrix}$$

**Q2.** Set up (but don't solve) the normal equations for fitting $y = w_0 + w_1 x + w_2 x^2$ to the data $(x, y) \in \{(1, 3), (2, 7), (3, 13), (4, 21)\}$. What is the shape of your $X$ matrix? (Hint: add a column for $x^2$.)

**Q3.** Two features are "temperature in Celsius" and "temperature in Fahrenheit" ($F = 32 + 1.8C$). If both are columns of $X$, what is $\text{rank}(X)$? What happens when you try to compute $(X^TX)^{-1}$? What is the practical fix?

**Q4. (Code)** Compare the Normal Equations solution to `jnp.linalg.lstsq` on the house-price data from the tutorial. Add a third (perfectly correlated) column to $X$ and observe the condition number of $X^TX$ blow up. Then add ridge regularisation $\lambda I$ to fix it.

> See `exercises/s4_least_squares.md` for full worked solutions and additional problems.

---

## Section 5 — Constrained Optimisation: Lagrange Multipliers (12 min)

### Why constrained optimisation?

So far we have minimised functions freely — gradient descent, normal equations. But many ML problems come with **constraints**: "minimise the loss *subject to* the weight vector satisfying some condition." You cannot just set the gradient to zero and stop; the unconstrained minimum may violate the constraint.

**Lagrange multipliers** are the tool for this. The idea: attach each constraint to the objective via a new scalar variable $\lambda$ (the **multiplier**), then solve the enlarged unconstrained system.

### The method

Given:

$$\min_{\mathbf{x}} \; f(\mathbf{x}) \quad \text{subject to} \quad g(\mathbf{x}) = 0$$

Form the **Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda \, g(\mathbf{x})$$

At a constrained optimum, the gradients of $f$ and $g$ must be parallel — one cannot improve $f$ without violating $g$. This is captured by setting all partial derivatives of $\mathcal{L}$ to zero:

$$\nabla_\mathbf{x} \mathcal{L} = 0 \qquad \text{and} \qquad \frac{\partial \mathcal{L}}{\partial \lambda} = 0$$

The second condition simply recovers the original constraint $g(\mathbf{x}) = 0$.

**Multiple constraints**: attach one multiplier per constraint:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_k \lambda_k g_k(\mathbf{x})$$

**Inequality constraints** ($g(\mathbf{x}) \leq 0$) lead to the **KKT conditions** (Karush-Kuhn-Tucker) — a generalisation where the multiplier must be non-negative and either the constraint is active or the multiplier is zero. SVMs use exactly this.

---

### Example 1 — Maximum Volume of a Box

**Problem**: find the maximum volume of a box with side lengths $u, v, w > 0$ subject to:

$$uv + uw + vw = \frac{c}{2}$$

Equivalently, $\max \; V = uvw$.

#### Step 1: Form the Lagrangian

Convert maximisation to minimisation by negating the objective:

$$\mathcal{L}(u, v, w, \beta) = -uvw + \beta\!\left(uv + uw + vw - \frac{c}{2}\right)$$

#### Step 2: First-order conditions

$$\frac{\partial \mathcal{L}}{\partial u} = -vw + \beta(v + w) = 0$$

$$\frac{\partial \mathcal{L}}{\partial v} = -uw + \beta(u + w) = 0$$

$$\frac{\partial \mathcal{L}}{\partial w} = -uv + \beta(u + v) = 0$$

$$\frac{\partial \mathcal{L}}{\partial \beta} = uv + uw + vw - \frac{c}{2} = 0$$

#### Step 3: Exploit symmetry

The constraint and objective are symmetric in $u, v, w$, so assume $u = v = w = x$. Substituting into any stationarity equation:

$$-x^2 + 2\beta x = 0 \quad \Rightarrow \quad \beta = \frac{x}{2}$$

#### Step 4: Apply the constraint

$$3x^2 = \frac{c}{2} \quad \Rightarrow \quad x = \sqrt{\frac{c}{6}}$$

#### Solution

$$u = v = w = \sqrt{\frac{c}{6}}$$

The optimal box is a **cube**. Maximum volume:

$$V_{\max} = \left(\frac{c}{6}\right)^{3/2}$$

---

### Example 2 — SVM Margin Maximisation (Week 4)

This is the direct ML application. **Hard-margin SVM** finds the hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$ that separates two classes with the largest possible margin.

**Primal problem**:

$$\min_{\mathbf{w},\, b} \; \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall i$$

The margin equals $\frac{2}{\|\mathbf{w}\|}$, so minimising $\|\mathbf{w}\|^2$ is equivalent to maximising the margin. The constraints say: every training point must be on the correct side with at least a margin of 1.

**Form the Lagrangian** (one multiplier $\alpha_i \geq 0$ per inequality constraint):

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i} \alpha_i \bigl[y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1\bigr]$$

**KKT stationarity conditions** ($\nabla_\mathbf{w}\mathcal{L} = 0$ and $\partial\mathcal{L}/\partial b = 0$):

$$\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i \qquad \text{(weight vector is a linear combination of training points)}$$

$$\sum_i \alpha_i y_i = 0$$

**Substituting back** into $\mathcal{L}$ eliminates $\mathbf{w}$ and $b$, giving the **dual problem** — maximise over $\boldsymbol{\alpha}$ only:

$$\max_{\boldsymbol{\alpha}} \; \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j)$$

$$\text{subject to} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

Key insight: the dual only involves dot products $\mathbf{x}_i \cdot \mathbf{x}_j$. Replace this with a kernel $k(\mathbf{x}_i, \mathbf{x}_j)$ and you get the **kernel SVM** — nonlinear boundaries for free, without ever explicitly computing high-dimensional features.

The **KKT complementary slackness** condition says $\alpha_i[y_i(\mathbf{w}\cdot\mathbf{x}_i + b) - 1] = 0$: either $\alpha_i = 0$ (point is not a support vector) or the constraint is exactly active (point sits on the margin). Only the support vectors matter for the decision boundary.

```python
import jax.numpy as jnp

# --- Verify box example numerically ---
c = 6.0
x_opt = jnp.sqrt(c / 6)
V_max = x_opt ** 3
constraint_val = 3 * x_opt ** 2

print(f"Optimal side length: {x_opt:.4f}")
print(f"Maximum volume:      {V_max:.4f}")
print(f"Constraint uv+uw+vw: {constraint_val:.4f}  (should be c/2 = {c/2})")

# --- Verify Lagrange multiplier ---
beta = x_opt / 2
stationarity = -x_opt**2 + 2 * beta * x_opt
print(f"Stationarity check:  {stationarity:.6f}  (should be 0)")

# --- SVM: illustrate dual dot products on a toy dataset ---
# Two linearly separable classes in 2D
X_pos = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 3.0]])   # label +1
X_neg = jnp.array([[-1.0, -1.0], [-2.0, -2.0], [-1.0, -3.0]])  # label -1

X_all = jnp.vstack([X_pos, X_neg])
y_all = jnp.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

# Gram matrix of dot products K_ij = x_i . x_j  (appears in SVM dual)
K = X_all @ X_all.T
print(f"\nGram (kernel) matrix shape: {K.shape}")
print(f"K:\n{K}")

# In the dual: maximise sum(alpha) - 0.5 * alpha^T (y_i y_j K_ij) alpha
# The matrix inside is the "kernel weighted by labels"
yy = jnp.outer(y_all, y_all)    # y_i * y_j for all pairs
Q = yy * K
print(f"\nQ matrix (y_i y_j K_ij):\n{Q}")
# The SVM solver maximises: sum(alpha) - 0.5 * alpha^T Q alpha
# subject to alpha >= 0, sum(alpha_i y_i) = 0
```

---

### Pattern to Remember

1. Write the objective $f(\mathbf{x})$ and the constraint(s) $g_k(\mathbf{x}) = 0$ (or $\leq 0$) explicitly
2. Form $\mathcal{L} = f + \sum_k \lambda_k g_k$ (one multiplier per constraint)
3. Set $\nabla \mathcal{L} = \mathbf{0}$ — one equation per variable plus one per multiplier
4. Exploit any symmetry to reduce the system
5. Substitute back into the constraint to find the optimal values
6. For inequalities, add $\lambda_k \geq 0$ and KKT complementary slackness $\lambda_k g_k = 0$

> **Course connection**:
> - **Week 4 (SVM I & II)**: the entire SVM derivation — primal, Lagrangian, dual, kernel trick — is built on the steps above. The $\alpha_i$ in the dual are the Lagrange multipliers.
> - **Week 5 (Logistic Regression)**: regularised logistic regression $\min_\mathbf{w} \text{loss} + \lambda\|\mathbf{w}\|^2$ is an unconstrained penalty form; the Lagrangian picture explains why $\lambda$ trades off fit vs. weight magnitude.
> - **Chapter 3.5 (Optimisation)** of the compendium covers KKT conditions, convexity, and constrained optimisation in full detail.

### Quick Exercises — Section 5

**Q1.** Maximise $f(x, y) = xy$ subject to $x + y = 10$. Form the Lagrangian, take partial derivatives, and find the optimal $(x^*, y^*)$. What is the maximum value? Does symmetry confirm it?

**Q2.** Find the point on the line $x + 2y = 5$ closest to the origin. Equivalently, minimise $x^2 + y^2$ subject to $x + 2y - 5 = 0$. Solve using Lagrange multipliers. Interpret geometrically.

**Q3.** Maximise entropy $H = -\sum_{i=1}^{n} p_i \ln p_i$ subject to $\sum_{i=1}^{n} p_i = 1$ and $p_i \geq 0$. Form the Lagrangian, set $\partial H / \partial p_i = 0$, and show the answer is the **uniform distribution** $p_i = 1/n$. This is a fundamental result used in Naive Bayes and information theory.

**Q4.** Write out the KKT conditions for the soft-margin SVM primal:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Identify each constraint, its multiplier, and what the complementary slackness condition means physically (when is a point a support vector?).

> See `exercises/s5_lagrange_multipliers.md` for full worked solutions and additional problems.

---

## Section 6 — Covariance and Positive Definiteness (8 min)

### Covariance matrix

Given a dataset $X$ (with each row a sample, mean-centred), the **covariance matrix** is:

$$\Sigma = \frac{1}{m-1} X^T X$$

$\Sigma$ is a square $n \times n$ matrix where:
- Diagonal entry $\Sigma_{ii}$ = variance of feature $i$ (how spread out it is)
- Off-diagonal entry $\Sigma_{ij}$ = covariance between features $i$ and $j$ (how they move together)

### Positive semi-definite matrices

A symmetric matrix $\Sigma$ is **positive semi-definite (PSD)** if for every vector $\mathbf{v}$:

$$\mathbf{v}^T \Sigma \mathbf{v} \geq 0$$

Covariance matrices are always PSD. Here's why this matters:
- **PSD** $\Leftrightarrow$ all eigenvalues $\geq 0$ (no negative "variance" directions)
- **Positive definite (PD)** $\Leftrightarrow$ all eigenvalues $> 0$ $\Leftrightarrow$ invertible

The quadratic form $\mathbf{v}^T \Sigma \mathbf{v}$ appears everywhere in ML: it measures the Mahalanobis distance (used in GMM), the variance explained in a given direction (used in PCA), and the curvature of loss surfaces (Hessians in optimisation).

### Gaussian Mixture Models (Week 6)

A GMM models data as a mixture of $K$ Gaussian distributions. Each Gaussian component $k$ has:
- A mean vector $\boldsymbol{\mu}_k \in \mathbb{R}^n$
- A **covariance matrix** $\Sigma_k \in \mathbb{R}^{n \times n}$ (must be PSD)

The probability of a point $\mathbf{x}$ under component $k$ is:

$$\mathcal{N}(\mathbf{x};\, \boldsymbol{\mu}_k, \Sigma_k) \propto \exp\!\left(-\tfrac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \Sigma_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)\right)$$

The term $(\mathbf{x} - \boldsymbol{\mu}_k)^T \Sigma_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)$ is the **Mahalanobis distance** — a quadratic form that generalises Euclidean distance by accounting for feature correlations. This is exactly the $\mathbf{v}^T A \mathbf{v}$ structure.

```python
import jax.numpy as jnp

# --- Compute covariance matrix of a dataset ---
X = jnp.array([
    [2.0, 1.0],
    [3.0, 2.0],
    [5.0, 4.0],
    [6.0, 5.0],
    [7.0, 6.0],
])

# Mean-centre
X_centered = X - X.mean(axis=0)

# Covariance matrix
m = X.shape[0]
Sigma = (X_centered.T @ X_centered) / (m - 1)
print(f"Covariance matrix:\n{Sigma}")

# Check it's symmetric
print(f"Symmetric: {jnp.allclose(Sigma, Sigma.T)}")

# Check positive definiteness: all eigenvalues > 0?
eigenvalues = jnp.linalg.eigvalsh(Sigma)
print(f"Eigenvalues: {eigenvalues}  (all positive = PD)")

# Quadratic form v^T Sigma v >= 0 for all v
v = jnp.array([1.0, -0.5])
quad_form = v @ Sigma @ v
print(f"v^T Sigma v = {quad_form:.4f}  (should be >= 0)")

# --- Mahalanobis distance ---
mu = X.mean(axis=0)
Sigma_inv = jnp.linalg.inv(Sigma)
point = jnp.array([4.0, 3.0])
diff = point - mu
mahal = jnp.sqrt(diff @ Sigma_inv @ diff)
eucl  = jnp.linalg.norm(diff)
print(f"\nEuclidean distance from mean: {eucl:.4f}")
print(f"Mahalanobis distance from mean: {mahal:.4f}")
```

> **Course connection**:
> - **Week 6 (GMM)**: the E-step computes $\mathcal{N}(\mathbf{x};\boldsymbol{\mu}_k, \Sigma_k)$ using the Mahalanobis quadratic form; the M-step updates $\Sigma_k = \sum_i r_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T / \sum_i r_{ik}$.
> - **Week 6 (Naive Bayes)**: assumes $\Sigma$ is diagonal (features are independent) — a massive simplification that makes the covariance trivially PD.

### Quick Exercises — Section 6

**Q1.** Compute the covariance matrix by hand for the dataset:

$$X = \begin{bmatrix}2 & 1\\4 & 3\\6 & 5\\8 & 7\end{bmatrix}$$

Is it positive definite? Check by computing its eigenvalues. What does the strong off-diagonal entry tell you about the two features?

**Q2.** For $\Sigma = \begin{bmatrix}4 & 0\\0 & 1\end{bmatrix}$, compute the Mahalanobis distance of points $\mathbf{p}_1 = (2, 0)$ and $\mathbf{p}_2 = (0, 2)$ from the origin. Both have the same Euclidean distance — why does Mahalanobis give different values? Which point is the "bigger surprise"?

**Q3.** A Naive Bayes classifier for class $C=1$ has learned: feature $x_1 \sim \mathcal{N}(3, 1)$ and feature $x_2 \sim \mathcal{N}(5, 4)$ (mean, variance). Write out the full covariance matrix $\Sigma$ that Naive Bayes is implicitly using. Why is it diagonal?

**Q4. (Code)** Generate 200 samples from a 2D Gaussian with mean $[2, 3]$ and covariance $\Sigma = [[3, 2],[2, 3]]$. Compute the sample covariance matrix. How close is it to the true $\Sigma$? Repeat with 2000 samples — observe convergence.

> See `exercises/s6_covariance.md` for full worked solutions and additional problems.

---

## Section 7 — Eigenvalues, SVD, and PCA (10 min)

### Eigenvectors and eigenvalues

An **eigenvector** of matrix $A$ is a special vector that does not change direction when $A$ is applied — only its length scales:

$$A\mathbf{v} = \lambda\mathbf{v}$$

$\lambda$ is the **eigenvalue** — the scaling factor. Most vectors rotate when multiplied by a matrix. Eigenvectors are the exceptions: they stay on the same line.

To find eigenvalues: solve the **characteristic polynomial** $\det(A - \lambda I) = 0$. The roots are the eigenvalues.

**Key facts for ML**:
- Symmetric matrices (like covariance matrices) always have **real eigenvalues** and **orthogonal eigenvectors**.
- A positive definite matrix has all positive eigenvalues.
- The trace equals the sum of eigenvalues; the determinant equals their product.

### Eigendecomposition

If a square matrix has $n$ linearly independent eigenvectors, it can be written as:

$$A = P D P^{-1}$$

where $D$ is diagonal (eigenvalues on the diagonal) and columns of $P$ are eigenvectors. For **symmetric** matrices, $P$ is orthogonal ($P^{-1} = P^T$), so:

$$A = P D P^T$$

This says: every symmetric transformation is just scaling along its eigenvector axes. No rotation in the eigenbasis — only stretching.

### SVD: the universal factorisation

**Singular Value Decomposition** (SVD) works for *any* matrix (not just square):

$$A = U \Sigma V^T$$

- $V^T$ ($n \times n$, orthogonal): rotates the input space
- $\Sigma$ ($m \times n$, diagonal): scales along orthogonal axes (singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)
- $U$ ($m \times m$, orthogonal): rotates the output space

Geometrically: every linear map, no matter how complex, is just rotate → scale → rotate. A sphere becomes an ellipse.

The singular values reveal importance: large $\sigma_i$ = direction that matters, small $\sigma_i$ = noise. The rank of $A$ equals the number of nonzero singular values.

### PCA from first principles (Week 9)

**Goal**: given a high-dimensional dataset, find the $k$ directions that capture the most variance, and project down to them.

**Steps**:

1. Mean-centre the data: $\tilde{X} = X - \bar{X}$
2. Compute the covariance matrix: $\Sigma = \frac{1}{m-1}\tilde{X}^T\tilde{X}$
3. Find the eigenvectors of $\Sigma$ — these are the **principal components**
4. Sort eigenvectors by their eigenvalues (largest first)
5. Project: $Z = \tilde{X} W_k$, where $W_k$ is the matrix of the top $k$ eigenvectors

**Why eigenvectors?** An eigenvector of $\Sigma$ is a direction $\mathbf{v}$ such that $\Sigma\mathbf{v} = \lambda\mathbf{v}$. The variance of the data projected onto $\mathbf{v}$ is:

$$\text{Var}(\tilde{X}\mathbf{v}) = \mathbf{v}^T \Sigma \mathbf{v} = \lambda \|\mathbf{v}\|^2 = \lambda$$

The eigenvalue $\lambda$ **is** the variance explained by that direction. Sorting by eigenvalue = sorting by explained variance.

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- Eigendecomposition of a 2x2 covariance matrix ---
Sigma = jnp.array([[3.0, 2.0],
                   [2.0, 2.0]])

eigenvalues, eigenvectors = jnp.linalg.eigh(Sigma)  # eigh for symmetric matrices
print("Eigenvalues (variance explained per PC):", eigenvalues[::-1])
print("Eigenvectors (principal components):\n", eigenvectors[:, ::-1])  # sort descending

# Verify: eigenvectors are orthogonal (dot product = 0)
v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]
print(f"v1 · v2 = {jnp.dot(v1, v2):.6f}  (should be ~0)")

# Reconstruct matrix from eigendecomposition: A = P D P^T
D = jnp.diag(eigenvalues)
Sigma_reconstructed = eigenvectors @ D @ eigenvectors.T
print(f"Reconstruction matches: {jnp.allclose(Sigma, Sigma_reconstructed)}")

# --- Full PCA on a small dataset ---
jnp_key = jnp.array([1.0])  # fix seed manually
X_raw = jnp.array([
    [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2],
    [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1],
    [1.5, 1.6], [1.1, 0.9],
])

# Step 1: mean-centre
X_c = X_raw - X_raw.mean(axis=0)

# Step 2: covariance matrix
m = X_raw.shape[0]
Cov = (X_c.T @ X_c) / (m - 1)
print(f"\nCovariance matrix:\n{Cov}")

# Step 3: eigendecomposition
vals, vecs = jnp.linalg.eigh(Cov)
# Sort descending (eigh returns ascending)
idx = jnp.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]
print(f"Eigenvalues: {vals}")
print(f"Variance explained: {vals / vals.sum() * 100} %")

# Step 4: project to 1D (top principal component)
W1 = vecs[:, :1]              # shape (2, 1): the first principal component
Z = X_c @ W1                  # shape (10, 1): projected data
print(f"\nProjected data (1D):\n{Z.flatten()}")

# Step 5: reconstruct for visualisation
X_reconstructed = Z @ W1.T + X_raw.mean(axis=0)

# --- SVD: k-rank approximation ---
A = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
               [9.0, 10.0, 11.0, 12.0]])

U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
print(f"\nSingular values: {S}")

for k in [1, 2, 3]:
    A_k = U[:, :k] @ jnp.diag(S[:k]) @ Vt[:k, :]
    err = jnp.linalg.norm(A - A_k, ord='fro')
    print(f"Rank-{k} approximation error (Frobenius): {err:.4f}")
```

> **Course connection**:
> - **Week 9 (PCA)**: the exact algorithm above. You project from $n$ features down to $k$ principal components, retaining the most variance. Eigenvalues tell you how many components to keep (scree plot, 90% explained variance threshold).
> - **Week 9 (Autoencoder)**: does the same as PCA but with a nonlinear encoder — the bottleneck layer learns a compressed representation. Linear autoencoders with MSE loss provably learn the same subspace as PCA.
> - **Week 10 (HMMs)**: the transition matrix $T$ where $T_{ij} = P(\text{state }j \mid \text{state }i)$ is a stochastic matrix. Its eigenvector with eigenvalue 1 is the stationary distribution.

### Quick Exercises — Section 7

**Q1.** Find the eigenvalues and eigenvectors of $A = \begin{bmatrix}4 & 1\\2 & 3\end{bmatrix}$ by hand. Verify by checking $A\mathbf{v} = \lambda\mathbf{v}$ for each pair. What is $\text{tr}(A)$ and $\det(A)$? Confirm they equal the sum and product of eigenvalues.

**Q2.** Perform PCA by hand on the four 2D points: $(1,2), (3,4), (5,4), (7,6)$. Mean-centre the data, compute the $2\times2$ covariance matrix, and find its eigenvectors. How much variance does the first PC explain?

**Q3.** The matrix $A = \begin{bmatrix}3 & 0\\0 & 0\end{bmatrix}$ has SVD $A = U\Sigma V^T$. Without computing, state the singular values, and describe what the transformation does geometrically (what shape does the unit circle map to?). What is the rank of $A$?

**Q4. (Code)** Apply PCA to the `sklearn` digits dataset (64 features, 10 classes). Plot the scree curve (explained variance vs. number of components). How many components are needed to retain 90% of variance? Project to 2D and plot coloured by digit class.

> See `exercises/s7_eigenvalues_pca.md` for full worked solutions and additional problems.

---

## Section 8 — Quick Reference for the ML Course (2 min)

| Week | Topic | Linear Algebra You Need |
|------|-------|------------------------|
| 1 | Perceptron | $\text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$ — dot product, hyperplane |
| 2 | Hinge Loss | $1 - y(\mathbf{w} \cdot \mathbf{x})$ — same dot product |
| 2 | Linear Regression | Normal equations: $\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$ |
| 3 | K-Means | Euclidean distance $\|\mathbf{x} - \boldsymbol{\mu}\|_2$, centroid = column mean |
| 3 | K-Medoids | L1 distance (more robust to outliers) |
| 4 | SVM | Lagrangian → KKT → dual: $\sum_i \alpha_i - \frac{1}{2}\sum_{ij}\alpha_i\alpha_j y_iy_j(\mathbf{x}_i\cdot\mathbf{x}_j)$ |
| 5 | Logistic Regression | $\sigma(\mathbf{w} \cdot \mathbf{x})$ — dot product through sigmoid |
| 5 | Neural Networks | $\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$ — matrix multiplication per layer |
| 6 | GMM | Mahalanobis: $(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$; covariance matrix update |
| 6 | Naive Bayes | Diagonal $\Sigma$ (independence assumption), per-feature variance |
| 8 | Decision Trees | Gini/entropy — scalar statistics, no matrices |
| 9 | PCA | Covariance matrix + eigendecomposition; project onto top-$k$ eigenvectors |
| 9 | Autoencoder | Same latent space as PCA but learned nonlinearly |
| 10 | HMM | Transition matrix $T$; forward-backward: matrix-vector products |
| 11 | MDP/RL | Value function $V$ is a vector; Bellman update: $V \leftarrow R + \gamma TV$ |

---

## Exercises

Work through these before the first ML lecture. Each maps to a concept you will use within the first two weeks.

**E1.** Given $\mathbf{w} = [2, -1, 0.5]$ and $\mathbf{x} = [1, 3, -2]$, compute $\mathbf{w} \cdot \mathbf{x}$ by hand. What is the Perceptron prediction (with $b = 0$)?

**E2.** Three 2D data points are $\mathbf{x}_1 = (1,1)$, $\mathbf{x}_2 = (4,5)$, $\mathbf{x}_3 = (3,2)$. Compute the centroid. Then compute the Euclidean distance from each point to the centroid — this is one iteration of K-Means with a single cluster.

**E3.** Solve the normal equations for the dataset:  
$X = \begin{bmatrix}1 & 1 \\ 1 & 2 \\ 1 & 3\end{bmatrix}$, $\mathbf{y} = \begin{bmatrix}2 \\ 4 \\ 5\end{bmatrix}$.  
Find $\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$. What line does this represent?

**E4.** Compute the covariance matrix of $X = \begin{bmatrix}1 & 2 \\ 3 & 4 \\ 5 & 6\end{bmatrix}$ (mean-centre first). Are the two features positively or negatively correlated? How can you tell from the matrix?

**E5.** For the $2 \times 2$ covariance matrix $\Sigma = \begin{bmatrix}4 & 2 \\ 2 & 3\end{bmatrix}$, find the eigenvalues. What percentage of variance does the first principal component explain?

**E6. (Code)** Implement PCA from scratch on the `sklearn` iris dataset (4 features, project to 2D). Plot the projected data coloured by class label. Compute the fraction of variance explained by each component.

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$, $\mathbf{w}$ | Vectors (bold lowercase) |
| $A$, $X$, $W$ | Matrices (uppercase) |
| $\|\mathbf{v}\|_2$ | Euclidean norm of $\mathbf{v}$ |
| $\mathbf{a} \cdot \mathbf{b}$ or $\mathbf{a}^T\mathbf{b}$ | Dot product |
| $A^T$ | Transpose of $A$ |
| $A^{-1}$ | Inverse of $A$ |
| $A^+$ | Pseudo-inverse of $A$ |
| $A = U\Sigma V^T$ | SVD factorisation |
| $A\mathbf{v} = \lambda\mathbf{v}$ | Eigenvector equation |
| $\Sigma$ | Covariance matrix |
| $\mathbf{v}^T\Sigma\mathbf{v}$ | Quadratic form (variance in direction $\mathbf{v}$) |

---

## Further Reading

All sections above have deeper treatments in the compendium:

- **Vectors**: Chapters 1.1–1.5 — Vector Spaces, Properties, Norms, Products, Basis
- **Matrices**: Chapters 2.1–2.5 — Properties, Types, Operations, Transformations, Decompositions
- **PCA / SVD**: Chapter 2.5 — Decompositions (eigendecomposition, SVD, PCA sections)
- **Probability for ML**: Chapter 4 (Statistics) and Chapter 5 (Probability) — needed for GMM, Naive Bayes, and HMMs
