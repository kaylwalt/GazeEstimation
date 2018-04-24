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
# x_lims = [int(min([float(x[0]) for x in data])) - 1, int(max([float(x[0]) for x in data])) + 1]
# y_lims = [int(min([float(x[1]) for x in data])) - 1, int(max([float(x[1]) for x in data])) + 1]
# plt.xlim(x_lims)
# plt.xlim(y_lims)

fig, ax = plt.subplots()
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     left='off',         # ticks along the top edge are off
#     labelbottom='off',
#     labelleft='off'
#     )

plt.scatter(*zip(*data))
plt.show()
