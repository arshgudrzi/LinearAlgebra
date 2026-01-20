#!/usr/bin/env python3
'''
Use eigenvalues to analyze stability of a simple system (population dynamics).

Example: Predator-Prey model
- Rabbits reproduce, get eaten by foxes
- Foxes need rabbits to survive

This is basic system stability analysis using eigenvalues.
NOT production code.
'''

import numpy as np

# Simple population dynamics matrix
# [rabbits_next]   [1.1  -0.05] [rabbits_now]
# [foxes_next  ] = [0.02  0.95] [foxes_now  ]
#
# 1.1: rabbits grow 10% per time step
# -0.05: foxes eat some rabbits
# 0.02: foxes benefit from eating rabbits
# 0.95: foxes decline 5% without enough food

A = np.array([
    [1.1, -0.05],
    [0.02, 0.95]
])

print("Population dynamics matrix:")
print(A)
print()

# Find eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
for i, val in enumerate(eigenvalues):
    print(f"  λ{i+1} = {val:.4f}")
print()

# Stability analysis:
# If ALL eigenvalues have absolute value < 1 → system stabilizes
# If ANY eigenvalue has absolute value > 1 → system explodes (unstable)
# If ANY eigenvalue has absolute value = 1 → equilibrium (marginal)

max_eigenvalue = np.max(np.abs(eigenvalues))
print(f"Maximum |eigenvalue|: {max_eigenvalue:.4f}")
print()

if max_eigenvalue < 1:
    print("STABLE: Population converges to equilibrium")
elif max_eigenvalue > 1:
    print("UNSTABLE: Population will explode or oscillate wildly")
else:
    print("MARGINAL: System at equilibrium")

# Simulate to verify
print("\n" + "="*50)
print("Simulation (verify our analysis):")
print("="*50)

population = np.array([100, 20])  # 100 rabbits, 20 foxes
print(f"Initial: Rabbits={population[0]:.0f}, Foxes={population[1]:.0f}")

for step in range(10):
    population = A @ population
    print(f"Step {step+1}: Rabbits={population[0]:.0f}, Foxes={population[1]:.0f}")
