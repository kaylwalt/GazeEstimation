import os
import numpy as np
import time
import scipy.io
import cv2
import random

data_folder = "../MPIIFaceGaze_kayl_norm"


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


for j in range(0,10):
    print("the random index this time is ", j)
    if j < 10:
        shufp = data_folder + "/p0{}_data.npy".format(j)
        shufl = data_folder + "/p0{}_labels.npy".format(j)
        print(shufp)
        shuffler_data = np.load(shufp)
        shuffler_labels = np.load(shufl)
    else:
        shufp = data_folder + "/p1{}_data.npy".format(j-10)
        shufl = data_folder + "/p1{}_labels.npy".format(j-10)
        print(shufp)
        shuffler_data = np.load(shufp)
        shuffler_labels = np.load(shufl)

    for i in range(0,15):

        if i == j:
            continue

        if i < 10:
            pathp = data_folder + "/p0{}_data.npy".format(i)
            pathl = data_folder + "/p0{}_labels.npy".format(i)
            print(pathp)
            train_data = np.load(pathp)
            train_labels = np.load(pathl).astype("float32")
            shuffler_data = np.concatenate((shuffler_data, train_data), axis=0)
            shuffler_labels = np.concatenate((shuffler_labels, train_labels), axis=0)
            shuffler_data, shuffler_labels = unison_shuffled_copies(shuffler_data, shuffler_labels)
            train_data = shuffler_data[int(len(shuffler_data)/2):,:,:,:]
            train_labels = shuffler_labels[int(len(shuffler_labels)/2):,:]
            shuffler_data = shuffler_data[0:int(len(shuffler_data)/2),:,:,:]
            shuffler_labels = shuffler_labels[0:int(len(shuffler_labels)/2),:]
            np.save(pathp, train_data)
            np.save(pathl, train_labels)


        else:
            pathp = data_folder + "/p1{}_data.npy".format(i-10)
            pathl = data_folder + "/p1{}_labels.npy".format(i-10)
            print(pathp)
            train_data = np.load(pathp)
            train_labels = np.load(pathl).astype("float32")
            shuffler_data = np.concatenate((shuffler_data, train_data), axis=0)
            shuffler_labels = np.concatenate((shuffler_labels, train_labels), axis=0)
            shuffler_data, shuffler_labels = unison_shuffled_copies(shuffler_data, shuffler_labels)
            train_data = shuffler_data[int(len(shuffler_data)/2):,:,:,:]
            train_labels = shuffler_labels[int(len(shuffler_labels)/2):,:]
            shuffler_data = shuffler_data[0:int(len(shuffler_data)/2),:,:,:]
            shuffler_labels = shuffler_labels[0:int(len(shuffler_labels)/2),:]
            np.save(pathp, train_data)
            np.save(pathl, train_labels)
