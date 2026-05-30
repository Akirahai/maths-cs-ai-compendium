# Exercises — Section 2: Similarity and Distance

> **Difficulty key**: ★ warm-up · ★★ core · ★★★ applied ML · ★★★★ challenge

---

## Conceptual Questions

**C1. ★** The dot product is $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$. What can you conclude about two vectors if their dot product is exactly zero? Give an ML example where this matters.

**C2. ★** Cosine similarity ignores magnitude; Euclidean distance does not. Give a scenario in text retrieval where cosine similarity is more appropriate than Euclidean distance. Give a scenario where Euclidean distance is more appropriate.

**C3. ★★** K-Means uses Euclidean distance; K-Medoids can use any distance metric. A dataset has many outliers. Which algorithm is more robust and why? What norm would you use in K-Medoids for outlier resistance?

**C4. ★★** A Perceptron with $\mathbf{w} = [1, 1]$ and $b = 0$ misclassifies $\mathbf{x} = [2, -3]$ (true label $+1$). The update rule is $\mathbf{w} \leftarrow \mathbf{w} + y\mathbf{x}$. What is the new $\mathbf{w}$? Did the score for $\mathbf{x}$ improve?

---

## Pencil-and-Paper Problems

**P1. ★** For each pair of vectors, compute the dot product, the angle between them, and state whether they are orthogonal, acute, or obtuse:

(a) $\mathbf{a} = [1, 0]$, $\mathbf{b} = [0, 1]$  
(b) $\mathbf{a} = [1, 2]$, $\mathbf{b} = [2, 1]$  
(c) $\mathbf{a} = [3, 4]$, $\mathbf{b} = [-4, 3]$  
(d) $\mathbf{a} = [1, 1, 1]$, $\mathbf{b} = [-1, -1, 2]$

**P2. ★** Compute the L1, L2, and L$\infty$ norms of:

(a) $\mathbf{v} = [3, -4, 0]$  
(b) $\mathbf{v} = [1, 1, 1, 1]$  
(c) $\mathbf{v} = [10, -1, 0, 0, 1]$

For (c), which norm is most sensitive to the large entry 10?

**P3. ★★** Compute pairwise cosine similarities for the three document vectors over vocabulary ["python", "java", "ML", "statistics"]:

$$\mathbf{d}_1 = [3, 0, 5, 1], \quad \mathbf{d}_2 = [0, 4, 3, 2], \quad \mathbf{d}_3 = [6, 0, 10, 2]$$

(a) Compute $\text{cos\_sim}(\mathbf{d}_1, \mathbf{d}_2)$, $\text{cos\_sim}(\mathbf{d}_1, \mathbf{d}_3)$, $\text{cos\_sim}(\mathbf{d}_2, \mathbf{d}_3)$.  
(b) Which two documents are most similar in direction? Are you surprised?  
(c) Compute Euclidean distances for the same pairs. Does the ranking agree with cosine similarity?  
(d) $\mathbf{d}_3 = 2\mathbf{d}_1$. What does cosine similarity give for two vectors where one is exactly twice the other?

**P4. ★★** A Perceptron classifies 2D points as $+1$ or $-1$. Weights: $\mathbf{w} = [2, -1]$, bias $b = 1$.

(a) Find the equation of the decision boundary (the line where $\mathbf{w}\cdot\mathbf{x} + b = 0$).  
(b) Classify: $(1, 3)$, $(2, 0)$, $(0, 0)$, $(-1, -1)$.  
(c) What is the perpendicular distance from the origin to the decision boundary? (Hint: it is $|b| / \|\mathbf{w}\|$.)  
(d) Point $(1, 3)$ is misclassified (true label $-1$). Apply one Perceptron update. What is the new $\mathbf{w}$?

**P5. ★★** One step of K-Means:

Initial centroids: $\boldsymbol{\mu}_1 = (1, 1)$, $\boldsymbol{\mu}_2 = (7, 5)$.

Data points: $(2, 1)$, $(3, 2)$, $(6, 4)$, $(8, 6)$, $(1, 3)$, $(9, 5)$.

(a) Assign each point to the nearest centroid (Euclidean distance).  
(b) Recompute the centroids as the mean of their assigned points.  
(c) Has the assignment changed after the update? Why or why not?

---

## Applied ML Problems

**A1. ★★** **Hinge Loss (Week 2)**. The hinge loss for a single training example is $\ell = \max(0,\; 1 - y(\mathbf{w}\cdot\mathbf{x}))$ where $y \in \{-1, +1\}$.

Given $\mathbf{w} = [1, 2]$, $b = 0$, for each point compute the hinge loss:

(a) $\mathbf{x}_1 = [2, 1]$, $y_1 = +1$  
(b) $\mathbf{x}_2 = [-1, 1]$, $y_2 = +1$  
(c) $\mathbf{x}_3 = [1, -2]$, $y_3 = -1$  
(d) $\mathbf{x}_4 = [-2, -1]$, $y_4 = -1$

Which points contribute to the loss? (Hint: a point contributes only if it is inside the margin or on the wrong side.)

**A2. ★★** **SVM Margin (Week 4)**. A linear SVM has weight vector $\mathbf{w} = [3, 4]$.

(a) Compute the margin $M = 2 / \|\mathbf{w}\|$.  
(b) If you double all weights ($\mathbf{w}' = 2\mathbf{w}$), does the decision boundary change? Does the margin change?  
(c) The SVM objective is to minimise $\frac{1}{2}\|\mathbf{w}\|^2$. Why the factor of $\frac{1}{2}$? (Hint: it simplifies the gradient.)  
(d) Points $\mathbf{x}_+ = [2, 1]$ (label $+1$) and $\mathbf{x}_- = [-2, -1]$ (label $-1$) are support vectors. Verify that $\mathbf{w}\cdot\mathbf{x}_+ = 1$ and $\mathbf{w}\cdot\mathbf{x}_- = -1$ (the margin constraint is active).

**A3. ★★★** **Distance sensitivity in clustering**. A 1D dataset has points: $\{1, 2, 3, 4, 100\}$.

(a) Compute the mean and the median.  
(b) K-Means with $K=1$ places its centroid at the mean. K-Medoids places it at the point that minimises the sum of distances. Find the K-Medoids centroid using L1 distance. Which is more robust to the outlier 100?  
(c) Now use L2 distance in K-Medoids. Does the centroid change?  
(d) Generalise: why is L1 distance ("sum of absolute deviations") robust to outliers while L2 is not?

**A4. ★★★** **Projection in the Perceptron**. The projection of $\mathbf{x}$ onto the weight vector $\mathbf{w}$ is:

$$\text{proj}_{\mathbf{w}}(\mathbf{x}) = \frac{\mathbf{x}\cdot\mathbf{w}}{\|\mathbf{w}\|^2}\mathbf{w}$$

(a) For $\mathbf{w} = [1, 0]$ and $\mathbf{x} = [3, 5]$, compute the projection. Describe geometrically what happened.  
(b) Show that the Perceptron score $\mathbf{w}\cdot\mathbf{x}$ equals $\|\mathbf{w}\| \cdot \|\text{proj}_{\mathbf{w}}(\mathbf{x})\|$. What does this say about the role of projection in classification?  
(c) Two points $\mathbf{x}_1 = [2, 3]$ and $\mathbf{x}_2 = [3, 2]$ have the same Euclidean norm. Do they get the same Perceptron score for $\mathbf{w} = [1, 2]$? Why not?

---

## Coding Exercises

**Code 1. ★** Implement and compare all three norms:

```python
import jax.numpy as jnp

def l1_norm(v):     # your code
def l2_norm(v):     # your code
def linf_norm(v):   # your code
def lp_norm(v, p):  # general Lp norm

# Demonstrate convergence: as p -> inf, Lp -> Linf
v = jnp.array([3.0, -4.0, 1.0, -2.0])
for p in [1, 2, 5, 10, 50, 100]:
    print(f"L{p}: {lp_norm(v, p):.6f}")
print(f"L∞: {linf_norm(v):.6f}")
```

**Code 2. ★★** Simulate one full K-Means iteration:

```python
import jax.numpy as jnp

def assign_clusters(X, centroids):
    """Assign each point in X to the nearest centroid. Return cluster labels."""
    # Hint: compute distance from each point to each centroid
    # Return array of shape (n_samples,) with values in {0, 1, ..., K-1}

def update_centroids(X, labels, K):
    """Recompute centroids as mean of assigned points."""

# Test on simple data
X = jnp.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
               [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])
centroids = jnp.array([[1.0, 1.0], [8.0, 8.0]])

labels = assign_clusters(X, centroids)
new_centroids = update_centroids(X, labels, K=2)
print(f"Labels: {labels}")
print(f"New centroids:\n{new_centroids}")
```

**Code 3. ★★★** Build a Perceptron from scratch and visualise the decision boundary evolving:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Linearly separable data
X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0],   # class +1
               [-1.0, -1.0], [-2.0, -2.0], [-1.0, -3.0]])  # class -1
y = jnp.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

# Implement the Perceptron algorithm:
# w = zeros(2), b = 0
# For each epoch (up to 100):
#   For each (x_i, y_i):
#     if y_i * (w . x_i + b) <= 0:   # misclassified
#       w += y_i * x_i
#       b += y_i
# After convergence, plot the data points and the decision boundary.
# Track total misclassifications per epoch and plot the learning curve.
```

---

## Solutions

<details>
<summary><strong>P1 Solutions</strong></summary>

(a) $\mathbf{a}\cdot\mathbf{b} = 0$ → **orthogonal** ($90°$)  
(b) $\mathbf{a}\cdot\mathbf{b} = 2+2 = 4 > 0$ → **acute** ($\theta = \arccos(4/\sqrt{5}\cdot\sqrt{5}) = \arccos(4/5) \approx 36.9°$)  
(c) $\mathbf{a}\cdot\mathbf{b} = -12+12 = 0$ → **orthogonal** ($90°$)  
(d) $\mathbf{a}\cdot\mathbf{b} = -1-1+2 = 0$ → **orthogonal** ($90°$)

</details>

<details>
<summary><strong>P3(b)(c)(d) Solution</strong></summary>

(b) $\mathbf{d}_1$ and $\mathbf{d}_3$ have cosine similarity exactly 1 — they point in the same direction because $\mathbf{d}_3 = 2\mathbf{d}_1$.  
(c) Euclidean distances rank differently because $\|\mathbf{d}_3 - \mathbf{d}_1\| = \|\mathbf{d}_1\| \neq 0$ even though they point the same way.  
(d) $\text{cos\_sim}(\mathbf{d}_1, \mathbf{d}_3) = \frac{\mathbf{d}_1 \cdot 2\mathbf{d}_1}{\|\mathbf{d}_1\|\cdot\|2\mathbf{d}_1\|} = \frac{2\|\mathbf{d}_1\|^2}{2\|\mathbf{d}_1\|^2} = 1$.

</details>

<details>
<summary><strong>A1 Solution</strong></summary>

(a) Score $= 1\cdot2 + 2\cdot1 = 4$. Hinge $= \max(0, 1-4) = 0$. No loss.  
(b) Score $= -1+2 = 1$. Hinge $= \max(0, 1-1) = 0$. Exactly on margin — no loss.  
(c) Score $= 1-4 = -3$. Hinge $= \max(0, 1-(-1)(-3)) = \max(0, 1-3) = 0$. No loss.  
(d) Score $= -2-2 = -4$. True label $-1$: $y \cdot \text{score} = 4$. Hinge $= \max(0, 1-4) = 0$. No loss.

All zero — this weight vector classifies everything correctly with margin. Try $\mathbf{w} = [0.5, 0.5]$ for non-zero losses.

</details>
