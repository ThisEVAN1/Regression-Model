import csv


FILE = 'scores_data.csv'
ALPHA = 0.00025


def main():
    x_values, y_values = read_file(FILE)
    weight = 0
    bias = 0
    for i in range(2000000):
        weight, bias = gradient_descent(weight, bias, x_values, y_values, ALPHA)
        if i % 100000 == 0:
            print(i, cost(weight, bias, x_values, y_values))
            print(weight, bias)
    


def read_file(file=FILE):
    '''Returns the x and y values'''
    x_values = []
    y_values = []
    with open(file, newline='') as f:
        data = csv.reader(f)
        header = next(data)
        for row in data:
            x_values.append(float(row[0]))
            y_values.append(float(row[1]))
    return x_values, y_values


def y_hat(weight, bias, x):
    '''Give the predicted y values'''
    return weight * x + bias


def cost(weight, bias, x_values, y_values):
    '''Find the cost of error'''
    total = 0

    for x, y in zip(x_values, y_values):
        total += (y_hat(weight, bias, x) - y) ** 2

    total /= 2 * len(x_values)

    return total


def compute_gradient(weight, bias, x_values, y_values, parameter, total):
    '''Return the computed gradient'''
    w = 0
    b = 0
    for x, y in zip(x_values, y_values):
        if parameter == 'weight':
            w += (y_hat(weight, bias, x) - y) * x
        elif parameter == 'bias':
            b += (y_hat(weight, bias, x) - y)
        else:
            raise ValueError('Invalid parameter')
        
    if parameter == 'weight':
        return w / total
    else:
        return b / total


def gradient_descent(weight, bias, x_values, y_values, alpha=ALPHA):
    total_values = len(x_values)
    w = weight
    b = bias

    w -= alpha * compute_gradient(weight, bias, x_values, y_values, 'weight', total_values)
    b -= alpha * compute_gradient(weight, bias, x_values, y_values, 'bias', total_values)
    return w, b


if __name__ == '__main__':
    main()