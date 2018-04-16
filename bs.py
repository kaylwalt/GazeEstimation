import os
import numpy as np
import time
import scipy.io

path = "../MPIIGaze/Data/6 points-based face model.mat"
#f = h5py.File(path, 'r')
f = scipy.io.loadmat(path)
cam = scipy.io.loadmat("../MPIIFaceGaze/p00/Calibration/Camera.mat")
print(cam)
#print(f)
#model = f['model']

#print(model.dtype)
#print(model.shape)

#np.save("./face_model.npy", model)
