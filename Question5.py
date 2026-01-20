#!/usr/bin/env python3
'''
Build a simple 2-layer neural network using only numpy.
Train it on XOR problem (classic non-linear problem).

This shows how backpropagation works mechanically.
NOT production code - just understanding the math.
'''

import numpy as np

# XOR dataset - classic test for neural networks
# XOR is NOT linearly separable (can't draw a straight line to separate)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("Training on XOR problem:")
print("Inputs:")
print(X)
print("Expected outputs:")
print(y)
print()

# Network architecture: 2 inputs -> 4 hidden -> 1 output

# Initialize weights randomly (small values)
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5  # input to hidden
b1 = np.zeros((1, 4))              # hidden bias
W2 = np.random.randn(4, 1) * 0.5  # hidden to output
b2 = np.zeros((1, 1))              # output bias

# Activation functions
def sigmoid(x):
    """Squashes values between 0 and 1"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip to avoid overflow

def sigmoid_derivative(x):
    """Derivative of sigmoid (needed for backprop)"""
    return x * (1 - x)

# Training parameters
learning_rate = 0.5
epochs = 10000

# Training loop
for epoch in range(epochs):
    # === FORWARD PASS ===
    # Layer 1
    z1 = X @ W1 + b1           # linear combination
    a1 = sigmoid(z1)           # activation (hidden layer output)
    
    # Layer 2
    z2 = a1 @ W2 + b2          # linear combination
    a2 = sigmoid(z2)           # activation (final output)
    
    # === COMPUTE LOSS ===
    loss = np.mean((y - a2) ** 2)  # mean squared error
    
    # === BACKWARD PASS (BACKPROPAGATION) ===
    # Output layer gradients
    d_loss_a2 = -2 * (y - a2)                    # derivative of loss w.r.t. output
    d_a2_z2 = sigmoid_derivative(a2)             # derivative of sigmoid
    d_loss_z2 = d_loss_a2 * d_a2_z2              # chain rule
    
    # Gradients for W2 and b2
    d_W2 = a1.T @ d_loss_z2                      # how much to change W2
    d_b2 = np.sum(d_loss_z2, axis=0, keepdims=True)
    
    # Hidden layer gradients (backpropagate through W2)
    d_loss_a1 = d_loss_z2 @ W2.T                 # how much hidden layer contributed
    d_a1_z1 = sigmoid_derivative(a1)
    d_loss_z1 = d_loss_a1 * d_a1_z1
    
    # Gradients for W1 and b1
    d_W1 = X.T @ d_loss_z1
    d_b1 = np.sum(d_loss_z1, axis=0, keepdims=True)
    
    # === UPDATE WEIGHTS (GRADIENT DESCENT) ===
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    
    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

print()
print("="*50)
print("TRAINING COMPLETE")
print("="*50)
print()

# Test the trained network
print("Final predictions:")
for i in range(len(X)):
    z1 = X[i:i+1] @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    prediction = sigmoid(z2)
    print(f"Input: {X[i]} | Expected: {y[i][0]} | Predicted: {prediction[0][0]:.4f}")

print()
print("Success! The network learned XOR (a non-linear function)")
print("This proves that matrix operations + non-linearity = universal function approximator")
