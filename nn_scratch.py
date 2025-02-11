import numpy as np
from sklearn.datasets import load_breast_cancer

# -------------------------------------------------------------
# 1) INITIALIZE PARAMETERS
# -------------------------------------------------------------
def init_params(layer_dims, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility
    
    params = {}
    L = len(layer_dims) - 1  # Number of layers
    
    for l in range(1, L+1):
        params[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) # Random weights
        params[f"b{l}"] = .1  # Bias initialized
    
    return params

# -------------------------------------------------------------
# 2) SIGMOID ACTIVATION FUNCTION
# -------------------------------------------------------------
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))  # Standard sigmoid function

# -------------------------------------------------------------
# 3) FORWARD PASS
# -------------------------------------------------------------
def forward_propagation(X, params):
    caches = {}
    A = X
    caches["A0"] = A  # Store input layer as "A0"
    
    L = sum(1 for k in params.keys() if k.startswith("W"))  # Count how many weight layers exist

    for l in range(1, L+1):
        W = params[f"W{l}"]  # Get weight for layer l
        b = params[f"b{l}"]  # Get bias for layer l
        Z = np.dot(W, A) + b  # Weighted sum of inputs + bias
        A = sigmoid(Z)  # Apply activation
        caches[f"Z{l}"] = Z  # Store pre-activation value
        caches[f"A{l}"] = A  # Store activation output
    
    return A, caches  # Final output + saved values

# -------------------------------------------------------------
# 4) COST FUNCTION
# -------------------------------------------------------------
def compute_cost(AL, Y):
    m = Y.shape[1]  # Number of training examples
    eps = 1e-15  # Small value to avoid log(0)
    return -np.sum(Y*np.log(AL+eps) + (1-Y)*np.log(1-AL+eps)) / m  # Binary cross-entropy loss

# -------------------------------------------------------------
# 5) BACKPROPAGATION
# -------------------------------------------------------------
def backward_propagation(params, caches, Y):
    grads = {}
    m = Y.shape[1]
    
    L = sum(1 for k in params.keys() if k.startswith("W"))  # Count layers

    AL = caches[f"A{L}"]  # Final activation output
    dZL = AL - Y  # Derivative of loss wrt ZL (output layer)

    A_prev = caches[f"A{L-1}"]
    grads[f"dW{L}"] = np.dot(dZL, A_prev.T) / m  # Gradient of weight
    grads[f"db{L}"] = np.sum(dZL, axis=1, keepdims=True) / m  # Gradient of bias

    for l in reversed(range(1, L)):
        dA = np.dot(params[f"W{l+1}"].T, dZL)  # Propagate gradient back
        Zl = caches[f"Z{l}"]
        Al = caches[f"A{l}"]
        dZl = dA * (Al * (1 - Al))  # Derivative of sigmoid
        A_prev_l = caches[f"A{l-1}"]

        grads[f"dW{l}"] = np.dot(dZl, A_prev_l.T) / m
        grads[f"db{l}"] = np.sum(dZl, axis=1, keepdims=True) / m
        
        dZL = dZl  # Move to previous layer
    
    return grads

# -------------------------------------------------------------
# 6) UPDATE PARAMETERS
# -------------------------------------------------------------
def update_params(params, grads, learning_rate=0.01):
    L = sum(1 for k in params.keys() if k.startswith("W"))  # Count layers
    
    for l in range(1, L+1):
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]  # Update weight
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]  # Update bias
    
    return params

# -------------------------------------------------------------
# 7) TRAIN FUNCTION
# -------------------------------------------------------------
def train_network(X, Y, layer_dims, epochs=1000, lr=0.15, seed=None):
    params = init_params(layer_dims, seed=seed)  # Initialize weights and biases
    
    for i in range(epochs):
        AL, caches = forward_propagation(X, params)  # Forward pass
        cost = compute_cost(AL, Y)  # Compute loss
        grads = backward_propagation(params, caches, Y)  # Compute gradients
        params = update_params(params, grads, learning_rate=lr)  # Update weights and biases
        
        if i % 10000 == 0:
            print(f"Epoch {i:4d}   Cost = {cost:.6f}")
    
    return params

# -------------------------------------------------------------
# RUNNING THE NETWORK
# -------------------------------------------------------------
if __name__ == "__main__":
    # Load Breast Cancer dataset from sklearn
    data = load_breast_cancer()
    
    # X contains features (each column is a sample) → Shape: (30, samples)
    X = data.data.T  
    
    # Y contains binary labels (1 = benign, 0 = malignant) → Shape: (1, samples)
    Y = data.target.reshape(1, -1)  

    # Normalize features (this makes training easier & prevents large values from dominating)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # Split dataset: 80% training, 20% testing
    m = X.shape[1]  # Total number of samples
    split_idx = int(0.8 * m)  # Get index for 80% of data

    # First 80% of samples for training, last 20% for testing
    X_train, X_test = X[:, :split_idx], X[:, split_idx:]
    Y_train, Y_test = Y[:, :split_idx], Y[:, split_idx:]

    # Define neural network structure:
    # - 30 input features (from dataset)
    # - 10 neurons in first hidden layer
    # - 5 neurons in second hidden layer
    # - 1 neuron in output layer (sigmoid activation for binary classification)
    layer_dims = [X_train.shape[0], 10, 5, 1]  

    # Train neural network on training data (5000 iterations, learning rate 0.01)
    params_final = train_network(X_train, Y_train, layer_dims, epochs=100000, lr=0.01, seed=1)

    # Run forward propagation on the test set
    AL_test, _ = forward_propagation(X_test, params_final)

    # Convert probabilities (AL_test) into 0 or 1 predictions
    preds_test = (AL_test >= 0.5).astype(int)  

    # Compute Accuracy, Precision, and Recall manually
    TP = np.sum((preds_test == 1) & (Y_test == 1))  # True Positives
    FP = np.sum((preds_test == 1) & (Y_test == 0))  # False Positives
    FN = np.sum((preds_test == 0) & (Y_test == 1))  # False Negatives
    TN = np.sum((preds_test == 0) & (Y_test == 0))  # True Negatives

    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Correct predictions / total samples
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0  # TP / (TP + FP)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0  # TP / (TP + FN)

    # Print results
    print("Test Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

