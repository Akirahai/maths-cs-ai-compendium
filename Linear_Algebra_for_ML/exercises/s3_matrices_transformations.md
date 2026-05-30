# Exercises — Section 3: Matrices and Transformations

> **Difficulty key**: ★ warm-up · ★★ core · ★★★ applied ML · ★★★★ challenge

---

## Conceptual Questions

**C1. ★** A matrix $W$ of shape $(128, 512)$ is a weight matrix in a neural network. What are the input and output dimensions of this layer? In which direction does information flow — left-to-right or right-to-left?

**C2. ★★** Matrix multiplication is not commutative ($AB \neq BA$). Give a concrete neural network example where swapping the order of two weight matrices produces completely different output dimensions (i.e., the multiplication becomes illegal).

**C3. ★★** A matrix has determinant 0. What does this mean geometrically? Why can you not use this matrix as a layer in a network if you need to reconstruct the original input (as in an autoencoder)?

**C4. ★★** The identity matrix $I$ is the multiplicative identity: $AI = IA = A$. What transformation does $I$ represent? What is its determinant? What are its eigenvalues?

---

## Pencil-and-Paper Problems

**P1. ★** Compute the following products, or state why they are impossible:

(a) $\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}5\\6\end{bmatrix}$

(b) $\begin{bmatrix}1&0&2\\0&1&1\end{bmatrix}\begin{bmatrix}3\\1\\2\end{bmatrix}$

(c) $\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}1&0\\0&1\end{bmatrix}$

(d) $\begin{bmatrix}1\\2\\3\end{bmatrix}\begin{bmatrix}4&5&6\end{bmatrix}$ (outer product)

(e) $\begin{bmatrix}1&2&3\end{bmatrix}\begin{bmatrix}4\\5\\6\end{bmatrix}$ (inner product)

**P2. ★** For the matrix $A = \begin{bmatrix}3 & 1\\ 2 & 4\end{bmatrix}$:

(a) Compute $\det(A)$  
(b) Compute $A^{-1}$ using the formula $A^{-1} = \frac{1}{\det(A)}\begin{bmatrix}d & -b\\-c & a\end{bmatrix}$  
(c) Verify $AA^{-1} = I$  
(d) Compute $\text{tr}(A)$ and the rank of $A$

**P3. ★★** The 2D rotation matrix is $R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\\sin\theta & \cos\theta\end{bmatrix}$.

(a) Compute $R(90°)$ explicitly.  
(b) Apply $R(90°)$ to $\mathbf{v} = [3, 1]^T$. Plot the original and rotated vector.  
(c) Verify that $\|R\mathbf{v}\| = \|\mathbf{v}\|$ (rotations preserve length).  
(d) Compute $R(90°)R(90°)$. What rotation does this correspond to? Verify with $R(180°)$.  
(e) Show that $R(\theta)^{-1} = R(\theta)^T = R(-\theta)$.

**P4. ★★** Apply each transformation to the unit square (corners at $(0,0), (1,0), (1,1), (0,1)$):

(a) Scaling: $S = \begin{bmatrix}2 & 0\\0 & 3\end{bmatrix}$  
(b) Reflection across y-axis: $\begin{bmatrix}-1 & 0\\0 & 1\end{bmatrix}$  
(c) Shear: $\begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}$  
(d) For each, compute the determinant and verify it equals the signed area of the transformed unit square.

**P5. ★★** Solve the following by row reduction (Gaussian elimination):

(a) $\begin{bmatrix}1&2&|&5\\3&4&|&11\end{bmatrix}$

(b) $\begin{bmatrix}2&-1&1&|&8\\-3&-1&2&|&-11\\-2&1&2&|&-3\end{bmatrix}$

(c) $\begin{bmatrix}1&2&3&|&6\\2&4&6&|&12\end{bmatrix}$ — What happens? How many solutions?

---

## Applied ML Problems

**A1. ★★** **Neural network forward pass (Week 5)**. A 2-layer network processes one input:

- Input: $\mathbf{x} = [1, 2, 3]^T$ (3 features)
- Layer 1 weights: $W_1 = \begin{bmatrix}0.1 & 0.2 & -0.1\\0.3 & -0.1 & 0.2\end{bmatrix}$, bias $\mathbf{b}_1 = [0.1, -0.1]^T$
- Activation: ReLU, i.e. $\text{ReLU}(z) = \max(0, z)$
- Layer 2 weights: $W_2 = \begin{bmatrix}0.5 & -0.5\end{bmatrix}$, bias $b_2 = 0$

(a) Compute $\mathbf{z}_1 = W_1\mathbf{x} + \mathbf{b}_1$.  
(b) Compute $\mathbf{h}_1 = \text{ReLU}(\mathbf{z}_1)$.  
(c) Compute the output $\hat{y} = W_2\mathbf{h}_1 + b_2$.  
(d) Without the activation function ($\text{ReLU}$ replaced by identity), what is the output? Express the whole network as a single matrix-vector product $W_2 W_1 \mathbf{x}$. Why does stacking linear layers without activations collapse to a single linear transformation?

**A2. ★★** **Dimensionality and information flow**. A dataset has $m=1000$ samples and $n=20$ features. You pass it through three layers with weight matrices $W_1(64\times20)$, $W_2(32\times64)$, $W_3(10\times32)$.

(a) What are the shapes of the intermediate representations after each layer?  
(b) Is information lost at any layer? (Compare the rank of each weight matrix to its output dimension.)  
(c) The final layer produces 10 outputs. If this is a classifier, what are these 10 numbers? What operation usually follows (e.g., softmax)?  
(d) What is the total number of trainable parameters (weights + biases)?

**A3. ★★★** **Gram matrix and kernel trick (Week 4)**. Given a dataset $X$ ($m \times n$), the **Gram matrix** is $K = XX^T$ ($m \times m$), where $K_{ij} = \mathbf{x}_i \cdot \mathbf{x}_j$.

For:
$$X = \begin{bmatrix}1 & 0\\0 & 1\\1 & 1\end{bmatrix}$$

(a) Compute $K = XX^T$.  
(b) Interpret $K_{12}$: what is the geometric relationship between $\mathbf{x}_1$ and $\mathbf{x}_2$?  
(c) Compute $K' = \phi(X)\phi(X)^T$ where $\phi([x_1, x_2]) = [x_1^2, \sqrt{2}x_1x_2, x_2^2]$ (the degree-2 polynomial feature map). Verify that $K'_{ij} = (\mathbf{x}_i \cdot \mathbf{x}_j)^2$ — the polynomial kernel.  
(d) Why does this mean an SVM can use nonlinear boundaries without explicitly computing $\phi(\mathbf{x})$?

**A4. ★★★** **Linear regression as matrix transformation**. The linear regression prediction is $\hat{\mathbf{y}} = X\mathbf{w}$.

The hat matrix (projection matrix) is $H = X(X^TX)^{-1}X^T$.

For $X = \begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix}$:

(a) Compute $H$ explicitly.  
(b) Verify $H^2 = H$ (idempotent — projecting twice is the same as projecting once).  
(c) Verify $H^T = H$ (symmetric).  
(d) What is the geometric interpretation: $\hat{\mathbf{y}} = H\mathbf{y}$ projects $\mathbf{y}$ onto what subspace?

---

## Coding Exercises

**Code 1. ★★** Build a full neural network forward pass from scratch:

```python
import jax.numpy as jnp

def relu(z):
    return jnp.maximum(0, z)

def forward_pass(x, layers):
    """
    x: input vector of shape (n_in,)
    layers: list of (W, b) tuples, all but the last followed by ReLU
    Returns the final output vector.
    """
    h = x
    for i, (W, b) in enumerate(layers):
        z = W @ h + b
        h = relu(z) if i < len(layers) - 1 else z   # no activation on last layer
    return h

# Define a 3-layer network: 4 -> 8 -> 4 -> 2
import numpy as np
rng = np.random.default_rng(42)
layers = [
    (jnp.array(rng.normal(size=(8, 4)) * 0.1), jnp.zeros(8)),
    (jnp.array(rng.normal(size=(4, 8)) * 0.1), jnp.zeros(4)),
    (jnp.array(rng.normal(size=(2, 4)) * 0.1), jnp.zeros(2)),
]

x = jnp.array([1.0, 2.0, 3.0, 4.0])
output = forward_pass(x, layers)
print(f"Output: {output}")

# Task: run the same input through all layers WITHOUT ReLU activations.
# Show that the result is equivalent to a single matrix multiplication W_eff @ x.
# Compute W_eff = W3 @ W2 @ W1 and verify.
```

**Code 2. ★★** Visualise how different transformations warp a 2D grid:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_transformed_grid(A, title):
    """Apply matrix A to a 2D grid of points and plot before/after."""
    xs = jnp.linspace(-2, 2, 10)
    ys = jnp.linspace(-2, 2, 10)
    grid_points = jnp.array([[x, y] for x in xs for y in ys])
    transformed = (A @ grid_points.T).T

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(grid_points[:, 0], grid_points[:, 1], s=10)
    axes[0].set_title("Original"); axes[0].set_aspect("equal")
    axes[1].scatter(transformed[:, 0], transformed[:, 1], s=10, c="red")
    axes[1].set_title(f"After: {title}"); axes[1].set_aspect("equal")
    plt.tight_layout(); plt.show()

transformations = {
    "Rotation 45°": jnp.array([[jnp.cos(jnp.pi/4), -jnp.sin(jnp.pi/4)],
                                 [jnp.sin(jnp.pi/4),  jnp.cos(jnp.pi/4)]]),
    "Scale (2x, 0.5y)": jnp.array([[2.0, 0.0], [0.0, 0.5]]),
    "Shear": jnp.array([[1.0, 1.0], [0.0, 1.0]]),
    "Singular (rank 1)": jnp.array([[1.0, 2.0], [2.0, 4.0]]),
}

for name, A in transformations.items():
    print(f"{name}: det={jnp.linalg.det(A):.2f}")
    plot_transformed_grid(A, name)
```

---

## Solutions

<details>
<summary><strong>P1 Solutions</strong></summary>

(a) $[17, 39]^T$  
(b) $[3+0+4, 0+1+2]^T = [7, 3]^T$  
(c) $A \cdot I = A$ (identity property)  
(d) $3\times3$ matrix with $(i,j)$ entry $= \mathbf{u}_i \mathbf{v}_j$: $\begin{bmatrix}4&5&6\\8&10&12\\12&15&18\end{bmatrix}$  
(e) Scalar: $4+10+18 = 32$ (dot product)

</details>

<details>
<summary><strong>P2 Solution</strong></summary>

(a) $\det = 3 \cdot 4 - 1 \cdot 2 = 10$  
(b) $A^{-1} = \frac{1}{10}\begin{bmatrix}4 & -1\\-2 & 3\end{bmatrix} = \begin{bmatrix}0.4 & -0.1\\-0.2 & 0.3\end{bmatrix}$  
(c) Verify: $\begin{bmatrix}3&1\\2&4\end{bmatrix}\begin{bmatrix}0.4&-0.1\\-0.2&0.3\end{bmatrix} = \begin{bmatrix}1.2-0.2 & -0.3+0.3\\0.8-0.8 & -0.2+1.2\end{bmatrix} = I$ ✓  
(d) $\text{tr} = 7$, rank $= 2$ (full rank, since $\det \neq 0$)

</details>

<details>
<summary><strong>A1(d) Solution</strong></summary>

Without activations: $\hat{y} = W_2(W_1\mathbf{x} + \mathbf{b}_1) + b_2 = (W_2W_1)\mathbf{x} + W_2\mathbf{b}_1 + b_2$.

This is still a linear function of $\mathbf{x}$. No matter how many linear layers you stack, the composition is always another linear transformation. Nonlinear activations (ReLU, sigmoid, tanh) are what make deep networks capable of learning nonlinear functions — without them, a 10-layer network is no more powerful than a single layer.

</details>
