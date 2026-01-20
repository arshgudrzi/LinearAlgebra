# Linear Algebra from First Principles

Learning linear algebra the right way - from vectors to building actual systems. No bullshit, just math and code.

## What's This?

My notes on learning linear algebra from scratch. Not a textbook. Not a course. Just one person figuring out how the math behind ML/AI actually works.

**Goal:** Understand what happens when you call `.fit()` instead of treating it like magic.

## What's Covered

- **Vectors** - arrows with direction and magnitude
- **Dot Products** - measuring similarity (cosine similarity, attention mechanisms)
- **Matrices** - transforming data (rotation, scaling, neural network layers)
- **Eigenvalues/Eigenvectors** - finding what doesn't change (PCA, PageRank, system stability)
- **SVD** - decomposing any matrix (Netflix recommendations, image compression)

Each concept includes:
- Intuitive explanation (what it actually does)
- Code examples (numpy implementations)
- Real applications (not toy problems)

## Practice Problems

Six problems ranging from easy to hard:
1. Phishing email classifier using dot products
2. Rotation matrices from scratch
3. System stability analysis with eigenvalues
4. Collaborative filtering with SVD
5. PCA implementation (no sklearn)
6. Neural network from scratch (backprop by hand)

Solutions included. Try them yourself first.

## Why This Exists

Most ML engineers use libraries without understanding what's underneath. That's fine until something breaks or you need custom solutions.

These notes are for people who want to:
- Actually understand what their models are doing
- Debug when gradient descent won't converge
- Implement algorithms from papers
- Not be limited by what sklearn offers

## Philosophy

**"Knowing basics deeply IS being advanced."**

You don't need to memorize every theorem. You need to understand the core primitives well enough that you can learn anything else on demand.

Like hacking: you don't learn to exploit one specific service. You learn the tools, then apply them creatively.

## What's Next

Calculus. Because understanding derivatives is how you understand why neural networks actually train.

## Usage

Read the notes. Run the code. Solve the problems. Break things. Learn.

This isn't production code. This is learning code. If you ship this to prod without modification, that's on you.

---

Written while learning. Mistakes included. PRs welcome if you find errors.
