import csv


FILE = 'scores_data.csv'
ALPHA = 1


def main():
    x_values, y_values = read_file(FILE)
    for i in range(len(x_values)):
        pass


def read_file(file=FILE):
    '''Returns the x and y values'''
    x_values = []
    y_values = []
    with open(file, newline='') as f:
        data = csv.reader(f)
        header = next(data)
        for row in data:
            x_values.append(row[0])
            y_values.append(row[1])
    return x_values, y_values


def y_hat(weight, bias, x):
    '''Give the predicted y values'''
    return weight * x + bias


if __name__ == '__main__':
    main()