#!/usr/bin/env python3
'''
Implement PCA from scratch and use it to reduce dimensionality.

PCA finds the directions where data varies the most.
This is NOT production code - just understanding the algorithm.
'''

import numpy as np
import matplotlib.pyplot as plt

# Generate some 2D data that's correlated (so PCA will be useful)
np.random.seed(42)
n_samples = 100

# Create data that's mostly along a diagonal
x = np.random.randn(n_samples)
y = x + np.random.randn(n_samples) * 0.5  # y correlates with x plus some noise

data = np.column_stack([x, y])

print("Original data shape:", data.shape)  # (100, 2)
print()

# === PCA ALGORITHM ===

# Step 1: Center the data (subtract mean)
mean = data.mean(axis=0)
centered = data - mean

print("Step 1: Centered data")
print(f"  Original mean: {mean}")
print(f"  Centered mean: {centered.mean(axis=0)}")  # should be ~[0, 0]
print()

# Step 2: Compute covariance matrix
cov_matrix = np.cov(centered.T)  # .T because np.cov expects features in rows

print("Step 2: Covariance matrix")
print(cov_matrix)
print()

# Step 3: Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Step 3: Eigenvalues and eigenvectors")
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Eigenvectors (columns):")
print(eigenvectors)
print()

# Step 4: Sort by eigenvalue (largest first)
# The eigenvector with largest eigenvalue is the "most important" direction
idx = eigenvalues.argsort()[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]

print("Step 4: Sorted (principal components)")
print(f"  PC1 (most important): {eigenvectors_sorted[:, 0]}")
print(f"  PC2 (second most): {eigenvectors_sorted[:, 1]}")
print()

# Step 5: Project data onto principal components
# To reduce to 1D, project onto first principal component only
pc1 = eigenvectors_sorted[:, 0]
reduced_1d = centered @ pc1  # this is now 1D

print("Step 5: Dimensionality reduction")
print(f"  Original shape: {data.shape}")
print(f"  Reduced shape: {reduced_1d.shape}")
print()

# How much variance does PC1 capture?
explained_variance = eigenvalues_sorted / eigenvalues_sorted.sum()
print("Explained variance:")
print(f"  PC1: {explained_variance[0]*100:.1f}%")
print(f"  PC2: {explained_variance[1]*100:.1f}%")
print()

# === VISUALIZATION ===
plt.figure(figsize=(12, 5))

# Plot 1: Original data with principal components
plt.subplot(1, 2, 1)
plt.scatter(centered[:, 0], centered[:, 1], alpha=0.5)
plt.arrow(0, 0, eigenvectors_sorted[0, 0]*2, eigenvectors_sorted[1, 0]*2, 
          head_width=0.2, color='red', label='PC1')
plt.arrow(0, 0, eigenvectors_sorted[0, 1]*2, eigenvectors_sorted[1, 1]*2, 
          head_width=0.2, color='blue', label='PC2')
plt.xlabel('Feature 1 (centered)')
plt.ylabel('Feature 2 (centered)')
plt.title('Original 2D Data with Principal Components')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot 2: Reduced 1D data
plt.subplot(1, 2, 2)
plt.scatter(reduced_1d, np.zeros_like(reduced_1d), alpha=0.5)
plt.xlabel('PC1 (first principal component)')
plt.ylabel('(collapsed to 1D)')
plt.title(f'Reduced to 1D (captures {explained_variance[0]*100:.1f}% variance)')
plt.grid(True)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pca_visualization.png', dpi=150, bbox_inches='tight')
print("Visualization saved to pca_visualization.png")
