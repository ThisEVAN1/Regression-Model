import csv


FILE = 'scores_data.csv'
ALPHA = 1


def main():
    print(read_file(FILE))


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


if __name__ == '__main__':
    main()