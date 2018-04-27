#This code is original
#
#
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as plticker
from helper_fun import cam_unit_to_cm

def get_eye_array_in_cm(filename):
    data = []
    with open(filename) as csvfile:
        pointer = csv.reader(csvfile)
        for row in pointer:
            data.append([row[2], row[3]])
    data = data[1:]
    data = list(map(lambda x : [cam_unit_to_cm(float(x[0])), cam_unit_to_cm(float(x[1]))], data))
    return data



if __name__ == "__main__":
    files = ['data/eyeposition_1.csv', 'data/eyeposition_0.csv', 'data/eyeposition_3.csv', 'data/eyeposition_2.csv']
    points_arrays = list(map(get_eye_array_in_cm, files))

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Distance in Centimeters', fontsize=15)

    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            data = points_arrays[(2 * i)+j]
            npdata = np.array(data)
            avpoint = np.mean(npdata, axis=0)
            print(avpoint)
            if i == 0 and j == 0:
                col.scatter(*zip(*data), label="observed values")
                col.scatter(0,0,c="green", label="real value")
                col.scatter(avpoint[0], avpoint[1], c='red', label="average value")
            else:
                col.scatter(*zip(*data))
                col.scatter(0,0,c="green")
                col.scatter(avpoint[0], avpoint[1], c='red')

    fig.legend(loc='upper right')
    plt.show()
