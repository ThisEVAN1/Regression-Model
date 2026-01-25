import csv
import numpy as np
import time
import sys


FILE = sys.argv[1]
ALPHA = 0.00025


def main():
    time_start = time.time()
    x_values, y_values = read_file(FILE)
    weight = 0
    bias = 0
    for i in range(2000000):
        weight, bias = gradient_descent(weight, bias, x_values, y_values, ALPHA)
        if i % 100000 == 0:
            print(i, cost(weight, bias, x_values, y_values))
            print(weight, bias)
    time_end = time.time()
    print(f"Time: {1000*(time_end - time_start):.4f} ms ")


def read_file(file=FILE):
    """Returns the x and y values"""
    x_values = []
    y_values = []
    with open(file, newline="") as f:
        data = csv.reader(f)
        header = next(data)
        for row in data:
            x_values.append(float(row[0]))
            y_values.append(float(row[1]))
    return np.array(x_values), np.array(y_values)


def y_hat(weight, bias, x):
    """Give the predicted y values"""
    return weight * x + bias


def cost(weight, bias, x_values, y_values):
    """Find the cost of error"""
    length = x_values.shape[0]
    total = 0

    for i in range(length):
        val = np.dot(x_values[i], weight) + bias
        total += (val - y_values[i]) ** 2

    total /= 2 * length

    return total


def compute_gradient(weight, bias, x_values, y_values, total):
    """Return the computed gradient"""
    w = 0
    b = 0
    for x, y in zip(x_values, y_values):
        w += (y_hat(weight, bias, x) - y) * x
        b += y_hat(weight, bias, x) - y
    return w / total, b / total


def gradient_descent(weight, bias, x_values, y_values, alpha=ALPHA):
    total_values = x_values.shape[0]
    w = weight
    b = bias
    gradient = list(
        map(
            lambda x: x * alpha,
            compute_gradient(weight, bias, x_values, y_values, total_values),
        )
    )

    w -= gradient[0]
    b -= gradient[1]
    return w, b


if __name__ == "__main__":
    main()
