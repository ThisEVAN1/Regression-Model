import numpy as np
import time
import sys


FILE = sys.argv[1]
ALPHA = 1
ITERATIONS = 10000


def main():
    time_start = time.time()
    X, y = read_file(FILE)
    μ, σ = normalize(X)

    m, n = X.shape
    w = np.zeros(n)
    b = 0
    
    for i in range(ITERATIONS):
        w, b = gradient_descent(w, b, X, y, ALPHA)

        # Print the iteration, cost, weight, and bias 10 times
        if i % (ITERATIONS / 10) == 0:
            w_true = w / σ
            b_true = b - np.sum((w * μ) / σ)

            print(f'Iteration: {i} Cost: {find_cost(w, b, X, y)}')
            print(f'Weights: {w_true} Bias: {b_true}')

    # Print the final results
    w_true = w / σ
    b_true = b - np.sum((w * μ) / σ)
    print('\n\tFINAL RESULT\t\n')
    print(f'Weights: {w_true} Bias: {b_true}')
    print(f'Iteration: {ITERATIONS} Cost: {find_cost(w, b, X, y)}')

    time_end = time.time()
    print(f"Time: {1000*(time_end - time_start):.4f} ms ")


def read_file(file=FILE):
    """Returns the x and y values"""
    data = np.loadtxt(file, delimiter=',', skiprows=1)

    X = data[:, :-1] # A 2d array of all values besides the last
    y = data[:, -1] # A 1d array of the last values

    return X, y


def normalize(X):
    '''Normalize data using z-score normalization'''
    μ = X.mean(axis=0)
    σ = X.std(axis=0)

    X -= μ 
    X /= σ

    return μ, σ


def y_hat(w, b, X):
    """Give the predicted y values"""
    return X @ w + b


def find_cost(w, b, X, y):
    """Find the cost of error"""
    m = X.shape[0]
    errors = y_hat(w, b, X) - y
    cost = (errors @ errors) / (2 * m)
    return cost


def compute_gradient(w, b, X, y):
    """Return the computed gradient"""
    m = X.shape[0]
    errors = y_hat(w, b, X) - y

    dw = (X.T @ errors) / m
    db = errors.sum() / m

    return dw, db


def gradient_descent(w, b, X, y, alpha=ALPHA):
    dw, db = compute_gradient(w, b, X, y)

    w -= alpha * dw
    b -= alpha * db

    return w, b


if __name__ == "__main__":
    main()
