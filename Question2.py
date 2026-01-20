#!/usr/bin/env python3
'''
Build a 2x2 rotation matrix that rotates 45 degrees. Apply it to vector [1,0] and verify.

This is just to understand how rotation matrices work mechanically.
NOT production code - just learning the math.
'''

import numpy as np

# Rotation matrix formula: [cos(θ)  -sin(θ)]
#                          [sin(θ)   cos(θ)]
# For 45 degrees, both cos and sin are √2/2 ≈ 0.707

def rotation_matrix(angle_degrees):
    """Create rotation matrix for given angle"""
    theta = np.radians(angle_degrees)  # convert to radians
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

# Create 45-degree rotation matrix
R = rotation_matrix(45)
print("45-degree rotation matrix:")
print(R)
print()

# Apply to vector [1, 0] (pointing right)
v = np.array([1, 0])
v_rotated = R @ v

print(f"Original vector: {v}")
print(f"Rotated vector: {v_rotated}")
print()

# Verify: a 45-degree rotation of [1,0] should give [√2/2, √2/2] ≈ [0.707, 0.707]
expected = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
print(f"Expected result: {expected}")
print(f"Match? {np.allclose(v_rotated, expected)}")  # checks if arrays are close enough

# Visual check: the rotated vector should have equal x and y components
# and should have length 1 (rotation preserves length)
print(f"\nLength of original: {np.linalg.norm(v):.3f}")
print(f"Length of rotated: {np.linalg.norm(v_rotated):.3f}")
print("(rotation should preserve length)")
