#This code is original
#
#
import numpy as np
import cv2
import sys
data_folder = "../MPIIFaceGaze_kayl_norm_leave_one"

data = np.load(data_folder + "/p12_data.npy")
labels = np.load(data_folder + "/p12_labels.npy")
print(len(data))
print(len(labels))
index = int(sys.argv[1])
print(index)
print("label : ", labels[index,:])
cv2.imshow("iamge", data[index,:,:,:])
cv2.waitKey(0)
