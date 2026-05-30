# Exercises — Section 1: Vectors as Data

> **Difficulty key**: ★ warm-up · ★★ core · ★★★ applied ML · ★★★★ challenge

---

## Conceptual Questions

**C1. ★** A data scientist says "I represented each user as a 500-dimensional vector." What does each dimension likely represent? Give two concrete examples of what those 500 entries might be.

**C2. ★** Why does a vector space require closure under *both* addition and scalar multiplication? Give a concrete ML example where violating one of these would break an algorithm.

**C3. ★★** Your dataset contains two features: "temperature in Celsius" and "temperature in Kelvin" ($K = C + 273.15$). Are the two feature vectors linearly independent? What rank does your feature matrix have? What should you do?

**C4. ★★** The zero vector $\mathbf{0}$ is a valid member of every vector space. In a dataset of feature vectors, what does the zero vector represent? Is it a valid data point? When might it cause problems?

---

## Pencil-and-Paper Problems

**P1. ★** Given $\mathbf{a} = [3, 0, 4]$ and $\mathbf{b} = [-1, 2, 2]$:

(a) Compute $\mathbf{a} + \mathbf{b}$  
(b) Compute $2\mathbf{a} - 3\mathbf{b}$  
(c) Compute $\|\mathbf{a}\|$, $\|\mathbf{b}\|$, and $\|\mathbf{a} + \mathbf{b}\|$  
(d) Verify the triangle inequality: $\|\mathbf{a} + \mathbf{b}\| \leq \|\mathbf{a}\| + \|\mathbf{b}\|$

**P2. ★** Normalise each vector to obtain the corresponding unit vector:

(a) $\mathbf{v} = [3, 4]$  
(b) $\mathbf{v} = [1, 1, 1]$  
(c) $\mathbf{v} = [2, 0, 0, -2]$

Verify $\|\hat{\mathbf{v}}\| = 1$ in each case.

**P3. ★★** Determine whether the following sets of vectors are linearly independent. If dependent, express one as a combination of the others:

(a) $\{[1, 2], [2, 4]\}$  
(b) $\{[1, 0, 0],\; [0, 1, 0],\; [0, 0, 1]\}$  
(c) $\{[1, 1, 0],\; [0, 1, 1],\; [1, 0, 1]\}$  
(d) $\{[1, 2, 3],\; [2, 4, 6],\; [0, 1, 0]\}$

**P4. ★★** A student's academic profile is the vector $\mathbf{s} = [85, 70, 90, 60]$ (scores in Maths, English, Science, Art out of 100).

(a) Find the unit vector in the direction of $\mathbf{s}$.  
(b) Another student scores $[42, 35, 45, 30]$. Are these two students' profiles parallel? What does this mean?  
(c) A third student scores $[80, 80, 80, 80]$. Compute the mean of the three profiles.

**P5. ★★** A subspace must pass through the origin and be closed under addition and scalar multiplication. Determine whether each set is a subspace of $\mathbb{R}^2$:

(a) $\{(x, y) : y = 2x\}$ — all points on the line $y = 2x$  
(b) $\{(x, y) : y = 2x + 1\}$ — all points on the line $y = 2x + 1$  
(c) $\{(x, y) : x \geq 0\}$ — the right half-plane  
(d) $\{(0, 0)\}$ — just the origin

---

## Applied ML Problems

**A1. ★★** Text classification uses **bag-of-words** vectors. A vocabulary has 5 words: ["the", "cat", "sat", "dog", "mat"]. Represent these sentences as vectors (word counts):

- $S_1$: "the cat sat on the mat"
- $S_2$: "the dog sat on the mat"
- $S_3$: "the cat sat"

(a) Write out the three feature vectors.  
(b) Compute $S_1 + S_2$. Does the result correspond to any real sentence? What operation on documents does addition represent?  
(c) Compute $\frac{S_1 + S_2 + S_3}{3}$. Interpret this as a "mean document."  
(d) Are $S_1$ and $S_3$ linearly independent? Explain what this means about the information each sentence carries.

**A2. ★★** In a recommendation system, user preferences are represented as vectors over 4 movie genres: [Action, Comedy, Drama, Sci-Fi]. Two users have profiles:

$$\mathbf{u}_1 = [5, 1, 2, 4], \quad \mathbf{u}_2 = [4, 2, 1, 5]$$

(a) Normalise both profiles.  
(b) A new movie has a genre vector $\mathbf{m} = [3, 0, 0, 4]$ (3 parts Action, 4 parts Sci-Fi). Which user is this movie closer to (Euclidean distance)?  
(c) A "balanced" user who likes all genres equally has profile $\mathbf{u}_0 = [3, 3, 3, 3]$. Express $\mathbf{u}_1$ as $\mathbf{u}_0$ plus a "deviation vector" $\mathbf{d} = \mathbf{u}_1 - \mathbf{u}_0$. What does $\mathbf{d}$ tell you about User 1's preferences?

**A3. ★★★** In a neural network, the **embedding** of a word is a dense vector in $\mathbb{R}^d$. It is claimed that in good word embeddings: $\text{vec}(\text{"king"}) - \text{vec}(\text{"man"}) + \text{vec}(\text{"woman"}) \approx \text{vec}(\text{"queen"})$.

(a) Which vector space axiom makes this arithmetic legal?  
(b) If word embeddings are unit vectors (on the unit sphere), what is the geometric interpretation of "king − man + woman"? Does the result necessarily land on the unit sphere?  
(c) What does it mean for "king" and "queen" to have high cosine similarity but not be parallel?

---

## Coding Exercises

**Code 1. ★** Implement the following from scratch (no `linalg` shortcuts):

```python
import jax.numpy as jnp

def magnitude(v):
    """Euclidean magnitude of a vector."""
    # your code here

def normalise(v):
    """Return unit vector in direction of v. Handle zero vector gracefully."""
    # your code here

def are_parallel(a, b, tol=1e-6):
    """Return True if a and b are parallel (one is a scalar multiple of the other)."""
    # Hint: cross product is zero for parallel 2D/3D vectors
    # your code here

# Test
v = jnp.array([3.0, 4.0])
print(magnitude(v))        # expected: 5.0
print(normalise(v))        # expected: [0.6, 0.8]
print(are_parallel(jnp.array([1.0, 2.0]), jnp.array([3.0, 6.0])))   # True
print(are_parallel(jnp.array([1.0, 2.0]), jnp.array([3.0, 5.0])))   # False
```

**Code 2. ★★** Build a mini feature-engineering pipeline:

```python
import jax.numpy as jnp

# Dataset: 6 students, features = [hours_studied, sleep_hours, prev_grade, anxiety_score]
X = jnp.array([
    [5.0,  7.0, 75.0, 3.0],
    [8.0,  6.0, 85.0, 2.0],
    [2.0,  8.0, 60.0, 4.0],
    [10.0, 5.0, 90.0, 1.0],
    [3.0,  9.0, 70.0, 5.0],
    [6.0,  7.0, 80.0, 2.0],
])

# Task 1: compute the mean feature vector (centroid of the dataset)
# Task 2: compute the deviation of each student from the mean
# Task 3: which student deviates most from the average (largest L2 norm of deviation)?
# Task 4: are any two feature columns linearly dependent?
#          (hint: use jnp.linalg.matrix_rank on the feature matrix)
# Task 5: add a "dummy" column of all 1s (for the intercept in linear regression).
#          Does this change the rank? Why is this column always linearly independent
#          of the others?
```

**Code 3. ★★★** Visualise the effect of linear dependence on a dataset:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Generate 100 2D points
key = jnp.array([42])
X = jnp.array([[i * 0.1, i * 0.1 + jnp.sin(jnp.array(i * 0.3)) * 0.1]
               for i in range(100)])

# Task 1: Plot the data. What do you observe?
# Task 2: Compute the covariance matrix and its rank.
# Task 3: What rank would you expect for perfectly linearly dependent features?
# Task 4: Add random noise to the second column: X[:, 1] += noise.
#          How does the rank change? How does the covariance change?
# Task 5: Explain why low-rank datasets are problematic for linear regression.
```

---

## Solutions

<details>
<summary><strong>P1 Solution</strong></summary>

(a) $\mathbf{a} + \mathbf{b} = [2, 2, 6]$  
(b) $2\mathbf{a} - 3\mathbf{b} = [6, 0, 8] - [-3, 6, 6] = [9, -6, 2]$  
(c) $\|\mathbf{a}\| = \sqrt{9+0+16} = 5$, $\|\mathbf{b}\| = \sqrt{1+4+4} = 3$, $\|\mathbf{a}+\mathbf{b}\| = \sqrt{4+4+36} = \sqrt{44} \approx 6.63$  
(d) $6.63 \leq 5 + 3 = 8$ ✓

</details>

<details>
<summary><strong>P3 Solution</strong></summary>

(a) **Dependent**: $[2,4] = 2[1,2]$  
(b) **Independent**: standard basis vectors — no axis direction can be built from the others  
(c) **Independent**: check that $c_1[1,1,0] + c_2[0,1,1] + c_3[1,0,1] = [0,0,0]$ forces $c_1=c_2=c_3=0$  
(d) **Dependent**: $[2,4,6] = 2[1,2,3]$, so the second vector is redundant regardless of the third

</details>

<details>
<summary><strong>P5 Solution</strong></summary>

(a) **Subspace** ✓ — passes through origin, closed under addition and scaling  
(b) **Not a subspace** ✗ — $(0,1)$ satisfies $y=2x+1$ but $2(0,1) = (0,2)$ does not satisfy $y=2x+1$  
(c) **Not a subspace** ✗ — $(-1, 0)$ should be in it (scaling $(1,0)$ by $-1$), but $-1 \not\geq 0$  
(d) **Subspace** ✓ — trivially closed

</details>
