import os
import h5py
import numpy as np
import time

def createNP():
    data_folder = "../MPIIFaceGaze_normalized"
    pics = []
    labels = []

    for i in range(0,10):
        path = data_folder + "/p0{}.mat".format(i)
        print(path)
        f = h5py.File(path, 'r')
        norm_pics = np.array(f.get('Data/data'))
        norm_pics = norm_pics.astype('uint8')
        norm_labs = np.array(f.get('Data/label'))
        np.save(path[:-4] + "_pics", norm_pics)
        np.save(path[:-4] + "_labs", norm_labs)
        # pics.append(norm_pics)
        # labels.append(norm_labs)

    for i in range(0,5):
        path = data_folder + "/p1{}.mat".format(i)
        print(path)
        f = h5py.File(path, 'r')
        norm_pics = np.array(f.get('Data/data'))
        norm_pics = norm_pics.astype('uint8')
        norm_labs = np.array(f.get('Data/label'))
        np.save(path[:-4] + "_pics", norm_pics)
        np.save(path[:-4] + "_labs", norm_labs)
        # pics.append(norm_pics)
        # labels.append(norm_labs)

    # all_pics = np.concatenate(pics)
    # all_labels = np.concatenate(labels)
    # print(all_pics.shape)
    # print(all_labels.shape)
    # np.save("all_pics", all_pics)
    # np.save("all_labels", all_labels)

def loadinmem():
    data_folder = "../MPIIFaceGaze_normalized"
    pics = []
    labels = []

    for i in range(0,10):
        pathp = data_folder + "/p0{}_pics.npy".format(i)
        pathl = data_folder + "/p0{}_labs.npy".format(i)
        print(pathp)
        norm_pics = np.load(pathp)
        norm_labs = np.load(pathl)
        print("pic dtype: ", norm_pics.dtype)
        print("lab dtype: ", norm_labs.dtype)
        pics.append(norm_pics)
        labels.append(norm_labs)

    for i in range(0,5):
        pathp = data_folder + "/p1{}_pics.npy".format(i)
        pathl = data_folder + "/p1{}_labs.npy".format(i)
        print(pathp)
        norm_pics = np.load(pathp)
        norm_labs = np.load(pathl)
        print("pic dtype: ", norm_pics.dtype)
        print("lab dtype: ", norm_labs.dtype)
        pics.append(norm_pics)
        labels.append(norm_labs)

    print("done and sleeping for 20")
    time.sleep(20)

if __name__ == "__main__":
    createNP()
