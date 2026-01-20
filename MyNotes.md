# Linear Algebra

The grundlage for every neural network --> matrix multiplication is a part of it
- Image processing --> matrix operations
- Cryptography --> linear transformations mod n
- My entire AI automation runs on this or something

---

## LEVEL 1: Core Primitive - Vectors

**What is a vector?** A list of numbers, duh. In programming kind of an array like `[4, 9]`

For example if we want to put it into code:

```python
# declare x and y whatever number you want
v = np.array([x, y])  # and for your own conscious the np.array() converts list to numpy array (don't want the headache, it will be faster math for the computer)
```

### Practical Example

```python
import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([3, 4])  # just remember the first one is X and the second one is Y
v2 = np.array([1, 2])  # for example here 1 point to right or better say in the X direction and 2 points up

# Visualize the item
plt.figure(figsize=(6,6))  # create figure, size is 6x6 inches

# plt.arrow(start_x, start_y, delta_x, delta_y, head_width, color, label)
# Draws arrow from (start_x, start_y) to (start_x+delta_x, start_y+delta_y)

plt.arrow(0, 0, v1[0], v1[1], head_width=0.3, color='blue', label='v1')
plt.arrow(0, 0, v2[0], v2[1], head_width=0.3, color='red', label='v2')

# did you get the idea from it how to write it? it all just starts from the 0 of X grid and 0 of Y grid and the rest as you see it is self explainable.

plt.grid(True)
plt.xlim(-1, 5)  # from where to where should the X go
plt.ylim(-1, 5)  # duh

plt.legend()
plt.title("Vectors are just arrows from origin")  # duh
plt.show()  # display the plot

# some vectors operations

print(f"v1 + v2 = {v1 + v2}")  # element wise addition [3,4] + [1,2] = [4,6]
print(f"2*v1 = {2*v1}")  # scalar multiplication: 2 * [3,4] = [6, 8] --> duh

# np.linalg.norm() calculates vector length (magnitude). --~ linalg --> lin + alg
# for 2D: sqrt(x^2 + y^2) = sqrt(3^2 + 4^2) = sqrt(9 + 16) = √25 = 5 --> did you enjoy the latex show off at the end Arash?

print(f"Length of v1: {np.linalg.norm(v1)}")
```

---

## LEVEL 2: Dot Product - Measuring Similarity

**What is it?** It is just multiplying corresponding numbers and summing them up.

```python
v1 = np.array([3, 4])
v2 = np.array([1, 2])

dot_product = np.dot(v1, v2)  # = 3*1 + 4*2 = 11 --> did you get the idea? every column is multiplied to each other.
# or
dot_product = v1 @ v2  # same thing, but cleaner syntax, oh fuck I love math and computers.
```

### The Magic Insight

The dot product tells you how much two vectors point in the same direction.

If you normalize between -1 to 0 to 1, if one, they are in the same direction exactly, and guess the rest. 0 means perpendicular (orthogonal).

### Why do I care? What are the use cases?

Well this is literally how:
- **Cosine similarity** works (recommendation systems)
- **Attention mechanisms** work (transformers) --> how neural networks find out on which stuff to focus. It is kind of their actionable insights and valuable insights.
- You find similar documents and embeddings
- You detect anomalies. (low similarity = outlier)

```python
import numpy as np

word_king = np.array([0.8, 0.2, 0.1])    # "royal, masculine, adult"
word_queen = np.array([0.8, 0.1, 0.9])   # "royal, feminine, adult"
word_man = np.array([0.1, 0.2, 0.8])     # "not-royal, masculine, adult"
word_apple = np.array([0.0, 0.5, 0.5])   # "food stuff, random"
```

### Cosine Similarity Function

Formula is: `(v1 · v2) / (|v1| × |v2|)` --> this is the dot product. Kind of exactly `(|v1| · |v2|) cosθ`

```python
def similarity(v1, v2):
    dot_product = np.dot(v1, v2)

    # normalize by lengths
    # np.linalg.norm(v) = length of vector = √(sum of squares) --> np.linalg.norm(v1) = √(3^2 + 4^2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Quick difference reminder:
    # - Magnitude: single vector's length → √(x² + y²) → gives you ONE number (how long the arrow is)
    # - Dot Product: two vectors → x1*x2 + y1*y2 → gives you ONE number (how aligned they are)
    # - Normal summing: two vectors → [x1+x2, y1+y2] → gives you a NEW VECTOR (combining directions)
    # - Scalar multiplication: number * vector → [k*x, k*y] → gives you a NEW VECTOR (stretching/shrinking)

    # just to be sure that you understood:
    #   magnitude uses the Pythagoras rule in triangles to find out the chord of the triangle, which is the arrow itself, duh
    #   dot product is the θ of the cosine of the angle from the chord of the triangle. which means 0 means it is orthogonal and 1 means complete straight
    #   normal summing is normal summing, duh. --> this needs a little visual, but think of it, as just putting the other end of the both arrows to the forward of them with a line
    #   scalar is just to be honest eigenvalue haha, you will learn it later. just kidding, scalar, duhhhh. I will put the latex also here. A * V = λ * V

    # the function is not over yet
    # WHY divide? The dot product alone is affected by vector lengths (longer = bigger number).
    # Dividing by |v1|*|v2| cancels out the lengths, giving you JUST the cosine of the angle.
    # Result: always between -1 and 1, no matter how long/short the vectors are.
    # Math: v1·v2 = |v1|*|v2|*cosθ  -->  rearrange  -->  cosθ = (v1·v2) / (|v1|*|v2|)
    #     did you get it? magnitude of two arrows will be this multiplying. magnitude of one arrow would be? guess. √(sum of squares)
    # So we divide dot_product by magnitudes to isolate cosθ (the alignment). 1 = same, 0 = perpendicular, -1 = opposite. anomaly detection works like this
    return dot_product / magnitude_product
    # no offense, but I think I need to give you intuition arash:
    #     above would be: v11*v21 + v12*v22 / √(v11**2 + v12**2) * √(v21**2 + v22**2)

# Calculate similarities
# .3f means "format as float with 3 decimal places"
print(f"king-queen similarity: {similarity(word_king, word_queen):.3f}")
print(f"king-man similarity: {similarity(word_king, word_man):.3f}")
print(f"king-apple similarity: {similarity(word_king, word_apple):.3f}")
```

---

## LEVEL 3: Matrices - Vector Transformers

A matrix is just a function that eats vectors and spits out transformed vectors. Think about it like that game in Windows that the ball hit many walls and shit before going out of the place.

```python
import numpy as np

# this matrix rotates vectors 90 degrees counterclockwise
rotation_matrix = np.array([
    [0, -1],
    [1, 0]
])
v = np.array([1, 0])  # point to the right. ummm... duh
v_rotated = rotation_matrix @ v  # now points up
```

### Did you get the intuition above?

```
[[0, -1],
 [1,  0]]
```

Here's the mechanical view: when you multiply this by vector `[x, y]`:

```
[0, -1]   [x]   [0*x + -1*y]   [-y]
[1,  0] × [y] = [1*x +  0*y] = [ x]
```

```python
print(f"Original: {v}")
print(f"Rotated: {v_rotated}")  # [0, 1] - points up!
```

### Wait, but WHY does that matrix rotate things? Trigonometry time!

Ok so here's the deal. That rotation matrix isn't random numbers. It's actually built from **cosine and sine** of the rotation angle.

**The general rotation matrix for angle θ (theta):**

```
[cos(θ)  -sin(θ)]
[sin(θ)   cos(θ)]
```

For 90 degrees (π/2 radians):
- cos(90°) = 0
- sin(90°) = 1

So it becomes:
```
[0  -1]
[1   0]
```

Boom. That's exactly the matrix we used above!

### But wait, what ARE cos, sin, tan, and cot?

**Imagine a right triangle (you know, the one with a 90° corner):**

```
        /|
       / |
      /  |  opposite (the side across from angle θ)
     /   |
    /θ___|
   hypotenuse  adjacent (the side next to angle θ)
```

Now the definitions:

| Function | What it means | Formula | Memory trick |
|----------|---------------|---------|--------------|
| **sin(θ)** | "sine" | opposite / hypotenuse | **S**OH - **S**ine = **O**pp / **H**yp |
| **cos(θ)** | "cosine" | adjacent / hypotenuse | **C**AH - **C**osine = **A**dj / **H**yp |
| **tan(θ)** | "tangent" | opposite / adjacent | **T**OA - **T**an = **O**pp / **A**dj |
| **cot(θ)** | "cotangent" | adjacent / opposite | It's just 1/tan, the flip of tan |

**SOH CAH TOA** - that's how everyone remembers this. Say it like "soh-kah-toh-ah". You're welcome.

### Why does this matter for rotation?

When you rotate a point around the origin by angle θ:
- The new x-coordinate = original_x × cos(θ) - original_y × sin(θ)
- The new y-coordinate = original_x × sin(θ) + original_y × cos(θ)

And that's EXACTLY what matrix multiplication does:

```
[cos(θ)  -sin(θ)]   [x]   [x·cos(θ) - y·sin(θ)]
[sin(θ)   cos(θ)] × [y] = [x·sin(θ) + y·cos(θ)]
```

```python
import numpy as np

def rotation_matrix(angle_degrees):
    """Create a rotation matrix for given angle in degrees"""
    # np.radians() converts degrees to radians (computers prefer radians)
    # radians = degrees × (π/180)
    theta = np.radians(angle_degrees)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

# Try different angles!
v = np.array([1, 0])  # pointing right

for angle in [45, 90, 180, 270]:
    R = rotation_matrix(angle)
    rotated = R @ v
    print(f"{angle}°: {rotated}")

# Output:
# 45°:  [0.707, 0.707]  (pointing diagonal up-right)
# 90°:  [0, 1]          (pointing up)
# 180°: [-1, 0]         (pointing left)
# 270°: [0, -1]         (pointing down)
```

> Fun fact: cos and sin are literally just the x and y coordinates of a point on a unit circle. That's why they're so perfect for rotation. The unit circle is your friend. Google it if you haven't seen it, it's beautiful.

---

### Matrix Multiplication - The Full Breakdown

Ok I realize we've been using `@` for matrix multiplication but never properly explained it. Let's fix that.

**The Rule:** Row × Column, sum them up

For A × B = C:
- Take a **row** from A
- Take a **column** from B
- Multiply corresponding elements
- Sum them all
- That's one element of C

**Visual example:**

```
A = [1  2  3]      B = [7   8]
    [4  5  6]          [9  10]
                       [11 12]

(2×3 matrix)  ×  (3×2 matrix)  =  (2×2 matrix)
```

To get C[0,0]: take row 0 of A, column 0 of B:
```
[1, 2, 3] · [7, 9, 11] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
```

To get C[0,1]: take row 0 of A, column 1 of B:
```
[1, 2, 3] · [8, 10, 12] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64
```

To get C[1,0]: take row 1 of A, column 0 of B:
```
[4, 5, 6] · [7, 9, 11] = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139
```

To get C[1,1]: take row 1 of A, column 1 of B:
```
[4, 5, 6] · [8, 10, 12] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154
```

**Result:**
```
C = [ 58   64]
    [139  154]
```

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

C = A @ B  # or np.matmul(A, B) or np.dot(A, B) for 2D
print(C)
# [[ 58  64]
#  [139 154]]
```

### The Shape Rule - CRITICAL!

**For A × B to work:** A's columns must equal B's rows

```
A: (m × n)  ×  B: (n × p)  =  C: (m × p)
         ↑         ↑
         └─ must match ─┘
```

Examples:
- (2×3) × (3×4) = (2×4) ✓
- (3×2) × (2×5) = (3×5) ✓
- (2×3) × (4×2) = ERROR ✗ (3 ≠ 4)

```python
# This fails
A = np.array([[1, 2, 3],    # shape (2, 3)
              [4, 5, 6]])
B = np.array([[1, 2],       # shape (3, 2) - wait no, this is (4, 2)
              [3, 4],
              [5, 6],
              [7, 8]])      # 4 rows, not 3!

# A @ B  # Would give error: shapes (2,3) and (4,2) not aligned
```

### Matrix "Division" - There's No Such Thing (Kind Of)

You can't divide matrices like numbers. BUT you can multiply by the **inverse**.

**Division equivalent:** A ÷ B  →  A × B⁻¹

**What's an inverse?** A matrix B⁻¹ such that B × B⁻¹ = I (identity matrix)

```python
import numpy as np

B = np.array([[4, 7],
              [2, 6]])

# Find inverse
B_inv = np.linalg.inv(B)
print("B inverse:")
print(B_inv)

# Verify: B × B⁻¹ = Identity
print("\nB × B⁻¹ =")
print(B @ B_inv)  # Should be [[1, 0], [0, 1]] (identity)
```

### Solving Linear Equations with "Division"

If you have: **A·x = b** and want to find x

Then: **x = A⁻¹·b**

```python
# Example: Solve the system:
# 2x + 3y = 8
# 4x + 5y = 14

A = np.array([[2, 3],
              [4, 5]])
b = np.array([8, 14])

# Method 1: Using inverse
x = np.linalg.inv(A) @ b
print("Solution:", x)  # [1. 2.] meaning x=1, y=2

# Method 2: Better way (numerically stable)
x = np.linalg.solve(A, b)
print("Solution:", x)  # Same: [1. 2.]

# Verify: 2(1) + 3(2) = 8 ✓, 4(1) + 5(2) = 14 ✓
```

### When Does Inverse NOT Exist?

Not all matrices have inverses! A matrix is **singular** (no inverse) if:
- Determinant = 0
- Rows are linearly dependent (one row is a multiple of another)

```python
# This matrix has NO inverse
bad_matrix = np.array([[1, 2],
                       [2, 4]])  # row 2 = 2 × row 1

print("Determinant:", np.linalg.det(bad_matrix))  # 0 or very close to 0

# This will raise LinAlgError
# np.linalg.inv(bad_matrix)  # ERROR: Singular matrix
```

### Quick Inverse Formula for 2×2

For a 2×2 matrix:
```
A = [a  b]      A⁻¹ = (1/det) × [ d  -b]
    [c  d]                      [-c   a]

where det = ad - bc
```

**Example:**
```
A = [4  7]    det = 4×6 - 7×2 = 24 - 14 = 10
    [2  6]

A⁻¹ = (1/10) × [ 6  -7] = [ 0.6  -0.7]
               [-2   4]   [-0.2   0.4]
```

```python
A = np.array([[4, 7], [2, 6]])
print(np.linalg.inv(A))
# [[ 0.6 -0.7]
#  [-0.2  0.4]]  # Matches!
```

> **TL;DR**: Matrix multiplication = row×column dot products. Matrix "division" = multiply by inverse. Not all matrices have inverses (check if det ≠ 0).

---

### Why does this matter?

Welllllll.... every neural network layer is literally just:

```python
output = activation(W @ input + bias)

# this is matrix multiplication, @ transforms your input vector
```

### Simple Example

```python
import numpy as np

# ReLU activation function: if x < 0, output 0; else output x
# this introduced non-linearity (allows network to learn complex patterns or something like this haha)

def relu(x):
    # np.maximum(a, b) returns element wise maximum --> ummm.... let me just show it down here:
    # Example: maximum([-1, 2, -3], 0) = [0, 2, 0]
    # did you get it? from above function? if x is less than 0, then go and just gives zero. or just gives 0 if it is bigger than 0
    return np.maximum(0, x)

# I obviously learned with AI and this is the example that they always give and if you just give in the code they always understand it. I am kind of a copy cat haha

# Input: 3 features (e.g., age, income, credit_score)
x = np.array([25, 50000, 700])

# weight matrix: transforms 3 inputs to 5 hidden neurons
# np.random.randn(rows, cols) creates matrix with random numbers
# from standard normal distribution (mean=0, std=1)
# Shape (5, 3) means: 5 output neurons, each takes 3 inputs
W = np.random.randn(5, 3) * 0.01  # no worries with this 0.01 fella, he is just there to keep the weights small initially.

# np.zeros(n) creates array of n zeros
b = np.zeros(5)  # bias term for each of 5 neurons

# Forward pass - THIS IS LITERALLY A NEURAL NET LAYER
# @ is matrix multiplication operator (same as np.dot or np.matmul)
# W @ x means: each row of W multiplies x and sums
# Result: 5 numbers (one per neuron)
hidden = relu(W @ x + b)

print(f"Input shape: {x.shape}")       # .shape shows dimensions of array
print(f"Weight shape: {W.shape}")      # (5, 3) = 5 rows, 3 columns
print(f"Output shape: {hidden.shape}") # (5,) = 1D array with 5 elements
print(f"Hidden layer activations: {hidden}")
```

> Wait a minute, did you just see? You little brilliant scientist. You just made a neural network layer from scratch, jesus you are smart. I am proud of you the person who is reading this note.

---

## LEVEL 4: Eigenvalues - What Doesn't Change

### The Core Idea

Some special vectors, when you transform them with a matrix, only get scaled. Think of them just getting stretched.

```
A @ v = λ * v
↑       ↑
matrix  just scales v by λ
```

> I did not have the energy to write the 3 lines above, AI did it, but you can think of the credit for me. danke

These special vectors are **eigenvectors**, and the scaling factors are **eigenvalues**.

### Why do we even care?

- **PCA (dimensionality reduction)**: eigenvectors show directions of maximum variance. What does that mean? The bigger the eigenvectors are, we kind of get it, which directions vary more or whatever. Well to be honest, you have to do a little bit of research on yourself, this is just a loser's compact note haha.
- **PageRank**: dominant eigenvector = importance ranking (I used this for fun on my company's URLs to see which one has the biggest eigenvector) it may not have been a revolution, but good training with different tools from crawling with Screaming Frog or custom scripts (this I did), and then making all the stuff with matrix and stuff that we do in the following upcoming crazy stuff
- **Stability analysis**: eigenvalues tell you if system explodes or converges
- **Markov chains**: eigenvectors = steady states.

### Ok enough, examples, we are here to learn, not just do examples and bluffs.

But first, let's learn how to find eigenvalues and eigenvectors BY HAND. Because numpy is cool but you should understand what's happening under the hood.

### How to Find Eigenvalues and Eigenvectors BY HAND (Manual Method)

Remember the equation: **A·v = λ·v**

This means: "matrix A transforms vector v, and the result is just v scaled by λ"

**Step 1: Rewrite the equation**

```
A·v = λ·v
A·v - λ·v = 0
(A - λI)·v = 0
```

Where I is the identity matrix (1s on diagonal, 0s elsewhere).

**Step 2: For non-trivial solutions (v ≠ 0), the determinant must be zero**

```
det(A - λI) = 0
```

This is called the **characteristic equation**. Solve for λ to get eigenvalues.

**Step 3: Plug each eigenvalue back to find eigenvectors**

---

### Full Example: 2x2 Matrix

Let's find eigenvalues and eigenvectors of:

```
A = [4  2]
    [1  3]
```

**Step 1: Set up (A - λI)**

```
A - λI = [4  2] - λ[1  0] = [4-λ    2  ]
         [1  3]    [0  1]   [1    3-λ]
```

**Step 2: Find determinant and set = 0**

For a 2x2 matrix `[a b; c d]`, determinant = ad - bc

```
det(A - λI) = (4-λ)(3-λ) - (2)(1)
            = 12 - 4λ - 3λ + λ² - 2
            = λ² - 7λ + 10
            = 0
```

**Step 3: Solve the quadratic equation**

```
λ² - 7λ + 10 = 0
(λ - 5)(λ - 2) = 0

λ₁ = 5
λ₂ = 2
```

These are your **eigenvalues**!

**Step 4: Find eigenvector for λ₁ = 5**

Substitute back into (A - λI)·v = 0:

```
[4-5    2  ] [x]   [-1  2] [x]   [0]
[1    3-5] [y] = [1  -2] [y] = [0]
```

From first row: -x + 2y = 0  →  x = 2y

Pick y = 1, then x = 2.

**Eigenvector for λ=5: v₁ = [2, 1]** (or any scalar multiple like [4, 2])

**Step 5: Find eigenvector for λ₂ = 2**

```
[4-2    2  ] [x]   [2  2] [x]   [0]
[1    3-2] [y] = [1  1] [y] = [0]
```

From first row: 2x + 2y = 0  →  x = -y

Pick y = 1, then x = -1.

**Eigenvector for λ=2: v₂ = [-1, 1]**

---

### Let's verify with numpy

```python
import numpy as np

A = np.array([
    [4, 2],
    [1, 3]
])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)  # [5. 2.]
print("Eigenvectors (as columns):")
print(eigenvectors)
# [[ 0.894  -0.707]   ← these are normalized versions of [2,1] and [-1,1]
#  [ 0.447   0.707]]

# Verify: A @ v should equal λ * v
v1 = eigenvectors[:, 0]  # first eigenvector
lambda1 = eigenvalues[0]  # first eigenvalue

print("\nVerification for λ=5:")
print(f"A @ v1 = {A @ v1}")
print(f"λ1 * v1 = {lambda1 * v1}")  # should be the same!
```

### Quick Recipe for 2x2 Matrices

For matrix `A = [a b; c d]`:

1. **Characteristic equation**: λ² - (a+d)λ + (ad-bc) = 0
   - (a+d) is the **trace** (sum of diagonal)
   - (ad-bc) is the **determinant**

2. **Solve for λ** using quadratic formula:
   ```
   λ = [(a+d) ± √((a+d)² - 4(ad-bc))] / 2
   ```

3. **Find eigenvectors** by solving (A - λI)·v = 0 for each λ

### What about 3x3 and bigger?

For 3x3, you get a cubic equation. Possible but annoying by hand.
For bigger matrices... just use numpy. Life is short, haha.

```python
# 3x3 example - let numpy do the heavy lifting
B = np.array([
    [1, 2, 1],
    [0, 3, 2],
    [0, 0, 2]
])

vals, vecs = np.linalg.eig(B)
print("Eigenvalues:", vals)  # [1. 3. 2.]
```

> Pro tip: For triangular matrices (all zeros above or below the diagonal), eigenvalues are just the diagonal elements! Easy mode.

---

Now let's see eigenvalues in action with PageRank:

```python
import numpy as np

# Web graph: who links to whom
# Pages: A, B, C, D
# Adjacency matrix (1 = link exists, 0 = no link)
links = np.array([
    [0, 1, 1, 0],  # A links to B and C (row 0)
    [1, 0, 0, 1],  # B links to A and D (row 1)
    [1, 0, 0, 1],  # C links to A and D (row 2)
    [0, 1, 1, 0]   # D links to B and C (row 3)
])

# for your own intuition, every row is a page, and every column with the number one is the links that are on the page to another page or whatever. I kind of think that I made you more confused. but that is on you. I used arch btw, this was completely irrelevant, but I am seeking social validation through arch. duh.

# Convert to probability matrix (random surfer model)
# Each row should sum to 1 (probability distribution)
# links.sum(axis=1) sums each row: [2, 2, 2, 2]
# keepdims=True keeps it as column vector for broadcasting
# Result: divide each row by its sum
P = links / links.sum(axis=1, keepdims=True)
```

### Intuition for the probability matrix

To be honest, I think that you may need a little intuition for this one, because I did not understand it at first either.

**The problem:** For PageRank, you need probabilities, not just "yes/no". If you're on page A and it has 2 links, you have a 50% chance of clicking each one.

**The conversion:**

```
links = [0, 1, 1, 0]  ← Page A has 2 links (to B and C)
        [1, 0, 0, 1]  ← Page B has 2 links
        ...

links.sum(axis=1) = [2, 2, 2, 2]  ← count links per row

P = links / [2, 2, 2, 2]

P = [0, 0.5, 0.5, 0]  ← 50% chance to go to B, 50% to C
    [0.5, 0, 0, 0.5]  ← 50% to A, 50% to D
    ...
```

**Why `keepdims=True`?**

```
Without it:
links.sum(axis=1) = [2, 2, 2, 2]  ← shape (4,) - flat array

With it:
links.sum(axis=1, keepdims=True) = [[2],   ← shape (4,1) - column
                                    [2],
                                    [2],
                                    [2]]
```

> Did you get it? If not, you should just give it to Claude or DeepSeek or whatever LLM you use. Because for these kind of stuff, it is important to understand mechanically.
>
> Note to myself, I will probably forget this, I should also try to hassle with LLM to learn this!

```python
# Find eigenvector with eigenvalue 1 (steady state)
# np.linalg.eig(matrix) returns (eigenvalues, eigenvectors)
# For matrix P.T (transpose), find what vector stays same after transformation
eigenvalues, eigenvectors = np.linalg.eig(P.T)
```

### Ok, I think you need a little more intuition

**What is Transpose?**

Flip the matrix - rows become columns, columns become rows:

```
P = [[a, b, c],        P.T = [[a, d],
     [d, e, f]]               [b, e],
                              [c, f]]
```

Or visually: mirror along the diagonal.

**Why transpose for PageRank?**

Your matrix P is set up as:
- Row = the page you're ON
- Column = the page you GO TO
- `P[A][B] = 0.5` means "from A, 50% chance to go to B"

But PageRank asks: "How important is each page?"

A page is important if other pages link TO it. So we need to flip the question:
- Instead of "where can I go FROM here?"
- We need "where can I come TO here FROM?"

```
P   = "from row, go to column"  (outgoing)
P.T = "to row, come from column" (incoming)
```

The eigenvector of P.T with eigenvalue 1 gives you the steady state - after infinite random clicks, what fraction of time do you spend on each page? That's the PageRank.

### But wait a little more, because I would also be a little still confused

**Imagine you're at a party with 4 rooms (A, B, C, D)**

People randomly walk through doors to other rooms.

- Matrix P says: "If you're IN room A, which doors can you walk OUT of?"
  - From A: 50% chance exit to B, 50% chance exit to C

But PageRank asks: "Which room has the most people in it over time?"

To answer that, you don't care where people GO. You care where people COME FROM.

- Matrix P.T says: "If you're IN room A, which doors do people ENTER from?"
  - To A: people come from B (50% of B's exits) and from C (50% of C's exits)

**Real example for the pages:**

Page A is "important" if:
- Many pages link TO A
- AND those pages are also important (they pass their importance along)

```
P   → "A sends 50% of its visitors to B"     (A is giving)
P.T → "A receives 50% of B's visitors"       (A is receiving)
```

**PageRank = popularity contest**

You don't become popular by linking to others (that's P).
You become popular by others linking to YOU (that's P.T).

---

**One more way to see it:**

Think of each link as a pipe that flows "importance juice":
- P = the pipes going OUT of each page
- P.T = the pipes coming IN to each page

The eigenvector finds the equilibrium - how much juice pools at each page when the flow stabilizes. You need the incoming pipes (P.T) to calculate that.

> Got it? If not, give all these above text to AI and tell to make it even simpler and more intuitive. I am kind of selfish, I learned until here, so you are on your own haha.
>
> "Anyways, back to it" ; Negan from the Walking Dead or something, I don't care. I just wrote this because it came up to my mind.

```python
# Find eigenvector with eigenvalue 1 (steady state)
# np.linalg.eig(matrix) returns (eigenvalues, eigenvectors)
# For matrix P.T (transpose), find what vector stays same after transformation
eigenvalues, eigenvectors = np.linalg.eig(P.T)

# Get eigenvector for eigenvalue ≈ 1 (the steady state)
# np.argmax(array) returns index of maximum value
# np.abs() because eigenvalues can be complex numbers
idx = np.argmax(np.abs(eigenvalues))

# extract the corresponding eigenvector (column idx)
# np.abs() to handle potential complex numbers (shouldn't happen here)
pagerank = np.abs(eigenvectors[:, idx])  # --> to be honest, this needs a little coding intuition, if you don't know coding, that is not my problem to visualize this for you and right now it is 11 pm and I need to go to sleep or whatever. but you get the idea.

# Normalize so sum = 1 (convert to probability distribution)
pagerank = pagerank / pagerank.sum()

print("PageRank scores:")
for i, score in enumerate(pagerank):  # enumerate gives (index, value) pairs
    # chr(65+i) converts: 0→'A', 1→'B', 2→'C', 3→'D' (ASCII codes)
    print(f"  Page {chr(65+i)}: {score:.3f}")
```

> **Mind Blown**: You just implemented the core of Google's PageRank algorithm. Eigenvectors, baby. You can try it with SEO tools and try to find out which page on your website is probably the strongest. And maybe Google notices that too. I don't know, I have never done the research on this topic to be honest.

---

## LEVEL 5: SVD - The Ultimate Decomposition

**What is it?** Every matrix can be broken down into:

```
A = U Σ V^T
```

> Well I could not believe it and did not know where it came from, but we gotta take their word for it, right? Hell nah, deep dive friend.

### Hell yeah, let's deep dive.

**Where does A = UΣV^T come from?**

The core idea: ANY matrix transformation (stretch, rotate, skew, whatever) can be broken into 3 simple steps:

```
A = U    ×    Σ    ×    V^T
    ↑         ↑         ↑
  rotate    scale    rotate
  (after)   (axes)   (first)
```

**Think of it like this:**

No matter how weird your transformation is, it's secretly just:
1. **V^T**: Rotate to align with some "natural axes"
2. **Σ**: Stretch/squish along those axes (diagonal = simple scaling)
3. **U**: Rotate to final orientation

---

### But WHERE does it mathematically come from?

Here's the trick. Take any matrix A and do this:

```
A^T × A = some matrix
```

This always gives you a symmetric matrix. Symmetric matrices are special - they have:
- Real eigenvalues
- Orthogonal eigenvectors (perpendicular to each other)

These eigenvectors become **V** (the right singular vectors).

Similarly:

```
A × A^T = another symmetric matrix
```

These eigenvectors become **U** (the left singular vectors).

And **Σ**? The singular values are the square roots of the eigenvalues of A^T×A.

---

### Intuition with a metaphor

Imagine you have a weird funhouse mirror (matrix A) that distorts your reflection.

SVD says: "That weird mirror is actually just:
1. Turn your head (V^T)
2. Look in a simple stretchy mirror (Σ) - only stretches horizontally/vertically
3. Turn your head back differently (U)"

**Every weird distortion = rotations + simple axis stretching.**

---

### Why do we care?

The singular values in Σ are sorted biggest to smallest. The big ones = "important" directions. You can throw away small ones and keep a compressed approximation. This is how:
- Image compression works
- Recommendation systems work (Netflix)
- Noise reduction works
- Dimensionality reduction (like PCA) works

> Ummmm.. but let's get back to it. Because we are not going to be da Vinci, but rather some guy who can understand this stuff and implement. We are not going theoretical, but rather theoretical (because it is cool, duh) + practical.

**Where:**
- **U** = left singular vectors (output space patterns)
- **Σ** = singular values (importance/strength)
- **V^T** = right singular vectors (input space patterns)

### Why This Is Insane
- **Recommendation systems**: Netflix, Amazon - all use SVD
- **Image compression**: JPEG uses similar idea
- **Noise reduction**: Keep strong signals, drop weak ones
- **Latent features**: Discover hidden patterns in data

Let's just build Netflix recommender system and then send it as a project in our application form to them so that they know we are learning and listening and we are smart (probably).

```python
import numpy as np

# User-Movie ratings (5 users, 4 movies)
# 0 = haven't watched (missing data)
ratings = np.array([
    [5, 3, 0, 1],   # User 1 ratings
    [4, 0, 0, 1],   # User 2 ratings
    [1, 1, 0, 5],   # User 3 ratings
    [1, 0, 0, 4],   # User 4 ratings
    [0, 1, 5, 4],   # User 5 ratings
], dtype=float)  # dtype=float needed for NaN handling

# Fill missing values (0s) with column mean
# np.nan = "Not a Number" - special float value for missing data
ratings[ratings == 0] = np.nan  # Replace 0s with NaN

# np.nanmean() calculates mean ignoring NaN values
# axis=0 means: compute mean for each column (each movie)
col_mean = np.nanmean(ratings, axis=0)

# np.where() finds indices where condition is True
# Returns tuple of (row_indices, col_indices)
inds = np.where(np.isnan(ratings))  # Find all NaN positions

# np.take(array, indices) extracts elements at indices
# Fill NaNs with corresponding column means
ratings[inds] = np.take(col_mean, inds[1])  # inds[1] = column indices

# SVD decomposition
U, S, Vt = np.linalg.svd(ratings)

# some dirty trick
# Keep only top 2 features (compress/denoise)
# This finds the 2 most important hidden patterns
k = 2
U_k = U[:, :k]           # First k columns of U (user features)
S_k = np.diag(S[:k])     # Convert k singular values to diagonal matrix
Vt_k = Vt[:k, :]         # First k rows of V^T (movie features)

# Reconstruct ratings (predictions)
# Matrix multiplication: (5,2) @ (2,2) @ (2,4) = (5,4)
predicted_ratings = U_k @ S_k @ Vt_k

print("Original ratings:")
print(ratings.astype(int))  # .astype(int) converts float → integer for display
print("\nPredicted ratings:")
print(predicted_ratings)
print("\nWhat User 1 might rate Movie 3:", predicted_ratings[0, 2])
```

> Well you know, I kind of feel we jumped into SVD a little soon, but that is how it is. I am sure you are smart enough to googleeeeeee and LLMeeeeeeeeeee a little.

I used this for hidden pattern prediction for churn analysis in a dataset, it is effective!

---

### How to Compute SVD BY HAND (Manual Method)

Ok numpy is nice, but let's see how SVD actually works step by step.

Remember: **A = U Σ Vᵀ**

**The Algorithm:**

1. Compute **AᵀA** (this gives you the right singular vectors V)
2. Find eigenvalues and eigenvectors of AᵀA
3. The eigenvectors become columns of **V**
4. The square roots of eigenvalues become the **singular values** (diagonal of Σ)
5. Compute **U** using: U = A·V·Σ⁻¹

Let's do it!

### Manual SVD Example

```
A = [3  2]
    [2  3]
```

**Step 1: Compute AᵀA**

```
Aᵀ = [3  2]     (transpose: rows become columns)
     [2  3]

AᵀA = [3  2] × [3  2] = [3·3+2·2  3·2+2·3] = [13  12]
      [2  3]   [2  3]   [2·3+3·2  2·2+3·3]   [12  13]
```

**Step 2: Find eigenvalues of AᵀA**

```
det(AᵀA - λI) = 0
det([13-λ   12  ]) = (13-λ)(13-λ) - 12·12 = 0
    [12    13-λ ]

(13-λ)² - 144 = 0
169 - 26λ + λ² - 144 = 0
λ² - 26λ + 25 = 0
(λ - 25)(λ - 1) = 0

λ₁ = 25,  λ₂ = 1
```

**Step 3: Singular values = square roots of eigenvalues**

```
σ₁ = √25 = 5
σ₂ = √1 = 1

Σ = [5  0]
    [0  1]
```

**Step 4: Find eigenvectors (these become V)**

For λ = 25:
```
[13-25   12  ] [x]   [-12  12] [x]   [0]
[12    13-25] [y] = [12  -12] [y] = [0]

-12x + 12y = 0  →  x = y
```
Normalized: v₁ = [1/√2, 1/√2] ≈ [0.707, 0.707]

For λ = 1:
```
[13-1   12 ] [x]   [12  12] [x]   [0]
[12   13-1] [y] = [12  12] [y] = [0]

12x + 12y = 0  →  x = -y
```
Normalized: v₂ = [1/√2, -1/√2] ≈ [0.707, -0.707]

```
V = [ 0.707   0.707]
    [ 0.707  -0.707]
```

**Step 5: Compute U**

U = A·V·Σ⁻¹

```python
import numpy as np

A = np.array([[3, 2], [2, 3]])

# Our manual calculations
V = np.array([[0.707, 0.707], [0.707, -0.707]])
sigma = np.array([[5, 0], [0, 1]])
sigma_inv = np.array([[1/5, 0], [0, 1]])  # inverse of diagonal is just 1/each

U = A @ V @ sigma_inv
print("U from manual calculation:")
print(U)

# Verify with numpy
U_np, S_np, Vt_np = np.linalg.svd(A)
print("\nU from numpy:")
print(U_np)
print("Singular values:", S_np)  # [5. 1.] - matches!
```

> Note: U and V might have different signs than numpy's version. That's ok - eigenvectors can point either direction (v and -v are both valid eigenvectors).

---

### PCA (Principal Component Analysis) BY HAND

PCA is basically SVD but on centered data. It finds the directions where your data varies the most.

**The Steps:**

1. **Center the data** (subtract mean from each feature)
2. **Compute covariance matrix**
3. **Find eigenvectors** (these are your principal components!)
4. **Project data** onto top eigenvectors

### PCA Manual Example

Let's say we have 4 data points with 2 features:

```
Data = [[2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2]]
```

**Step 1: Center the data**

```
mean_x = (2.5 + 0.5 + 2.2 + 1.9) / 4 = 1.775
mean_y = (2.4 + 0.7 + 2.9 + 2.2) / 4 = 2.05

Centered = [[2.5-1.775, 2.4-2.05],   = [[ 0.725,  0.35 ],
            [0.5-1.775, 0.7-2.05],      [-1.275, -1.35 ],
            [2.2-1.775, 2.9-2.05],      [ 0.425,  0.85 ],
            [1.9-1.775, 2.2-2.05]]      [ 0.125,  0.15 ]]
```

**Step 2: Compute covariance matrix**

```
Cov = (1/(n-1)) × Centeredᵀ × Centered
```

For our 2D data:
- cov(x,x) = variance of x
- cov(x,y) = cov(y,x) = covariance between x and y
- cov(y,y) = variance of y

**Step 3: Find eigenvectors of covariance matrix**

These eigenvectors are your principal components! The eigenvalues tell you how much variance each component captures.

**Step 4: Project**

To reduce to 1D, project onto the first eigenvector (the one with largest eigenvalue).

```python
import numpy as np

# Our data
data = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2]
])

# Step 1: Center
mean = data.mean(axis=0)
centered = data - mean

print("Centered data:")
print(centered)

# Step 2: Covariance matrix
# np.cov expects features in rows, so we transpose
cov_matrix = np.cov(centered.T)
print("\nCovariance matrix:")
print(cov_matrix)

# Step 3: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:", eigenvalues)
print("Eigenvectors (columns):")
print(eigenvectors)

# Sort by eigenvalue (largest first)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 4: Project onto first principal component (reduce to 1D)
pc1 = eigenvectors[:, 0]  # first principal component
projected_1d = centered @ pc1

print("\nFirst principal component:", pc1)
print("Data projected to 1D:", projected_1d)

# Explained variance ratio
total_variance = eigenvalues.sum()
explained_ratio = eigenvalues / total_variance
print(f"\nPC1 explains {explained_ratio[0]*100:.1f}% of variance")
print(f"PC2 explains {explained_ratio[1]*100:.1f}% of variance")
```

### The Connection: SVD and PCA

Here's the cool part - PCA using SVD is more efficient:

```python
# PCA the SVD way (what sklearn does internally)
U, S, Vt = np.linalg.svd(centered)

# V^T rows are the principal components!
print("Principal components from SVD:")
print(Vt)

# Project using SVD
projected_svd = centered @ Vt.T[:, 0]  # first PC
print("Projected (SVD way):", projected_svd)
```

**Why use SVD for PCA?**
- Numerically more stable
- Don't need to compute covariance matrix explicitly
- Faster for high-dimensional data

> **TL;DR**: PCA = center data + find eigenvectors of covariance. SVD is just a faster/stabler way to compute PCA. Both find the "most important directions" in your data.

There is a test that I also solve tomorrow and will put it on GitHub in the same folder. Thanks for being with me.

---

> And remember, as in hacking, you don't learn to hack something specific. You just learn the tools and how they work. Your intuition and creativity is the weapon.
>
> So as in fighting sports you learn the moves. But you don't learn to beat some athlete. You just use the movements and refine them to beat the opponent. But I guess we all are just not into fighting sports if we are studying math haha. Tschüssi or something cuter.

tests : 
### Easy : 
1. Create vectors for 5 different phishing email features (suspicious links, urgency w    ords, etc.). Use dot product to classify new emails.
2. Build a 2x2 rotation matrix that rotates 45 degrees. Apply it to vector [1,0] and v    erify.
### Medium : 
3. Use eigenvalues to analyze stability of a simple system (e.g., population dynamics)    .
4. Buil a collaborative filtering System for your pentesting tool recommendations using SVD.
### hard : 
5. implement PCA from scratch and use it to reduce dimensionality of a dataset
6. Build a simple neural network (with 2 layers) using only numpy matrix operations. Train it on whatever
9. this one is optional and overkill to be honest. But you can also make a search engine if you want.

