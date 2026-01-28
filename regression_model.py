import numpy as np
import time
import sys


FILE = sys.argv[1]
ALPHA = 0.00025
ITERATIONS = 100000


def main():
    time_start = time.time()
    X, y = read_file(FILE)

    m, n = X.shape
    w = np.zeros(n)
    b = 0
    
    for i in range(ITERATIONS):
        w, b = gradient_descent(w, b, X, y, ALPHA)
        if i % 10000 == 0:
            print(i, find_cost(w, b, X, y))
            print(w, b)
    time_end = time.time()
    print(f"Time: {1000*(time_end - time_start):.4f} ms ")


def read_file(file=FILE):
    """Returns the x and y values"""
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    X = data[:, :-1] # A 2d array of all values besides the last
    y = data[:, -1] # A 1d array of the last values
    
    return X, y


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
