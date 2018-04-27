#This code is original
#
#
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as plticker



data = []
with open('data/eyeposition_2.csv') as csvfile:
    pointer = csv.reader(csvfile)
    for row in pointer:
        data.append([row[2], row[3]])
data = data[1:]

fig, ax = plt.subplots()
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.scatter(*zip(*data))
plt.show()
