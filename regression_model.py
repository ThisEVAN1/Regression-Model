import csv


FILE = 'scores_data.csv'
ALPHA = 1


def main():
    x_values, y_values = read_file(FILE)
    for i in range(len(x_values)):
        print(cost(1, 0, x_values, y_values))


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






if __name__ == '__main__':
    main()