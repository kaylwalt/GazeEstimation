import math
import numpy as np
import cv2

def t(vec):
    theta = math.asin(-vec[1])
    phi = math.atan2(-vec[0], -vec[2])

    return [theta, phi]



def trev(angles):
    """input as [theta, phi], [rot around x, rot around y]"""
    angles = [-angles[0], -angles[1]]
    rot_mat_x = np.array([[1, 0, 0],
                          [0, math.cos(angles[0]), -math.sin(angles[0])],
                          [0, math.sin(angles[0]), math.cos(angles[0])]])
    rot_mat_y = np.array([[math.cos(angles[1]), 0, -math.sin(angles[1])],
                          [0, 1, 0],
                          [math.sin(angles[1]), 0, math.cos(angles[1])]])
    start = np.array([[0], [0], [-1]])

    final_rot = np.matmul(rot_mat_x, rot_mat_y)
    answer = np.matmul(final_rot, start)
    
    return answer


images = np.load("../MPIIFaceGaze_normalized/p00_pics.npy")
labels = np.load("../MPIIFaceGaze_normalized/p00_labs.npy")

print(images.shape)
print(labels.shape)

images = np.swapaxes(np.swapaxes(images, 1, 3), 1, 2)
print(images.shape)

i1 = images[0]
l1 = labels[0]
print(l1)
print(trev(l1))

cv2.imshow("image1", i1)
cv2.waitKey(0)
