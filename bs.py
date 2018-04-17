import os
import numpy as np
import time
import scipy.io
import cv2

data = np.load("../MPIIFaceGaze_kayl_norm/p00_data.npy")
labels = np.load("../MPIIFaceGaze_kayl_norm/p00_labels.npy")
print(data.shape)
print(labels.shape)

index = 2779
cv2.imshow("image", data[index,:,:,:])
print("theta is ", labels[index, 0])
print("phi is ", labels[index, 1])
cv2.waitKey(0)
