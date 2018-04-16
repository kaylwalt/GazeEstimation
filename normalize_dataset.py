import os
import h5py
import numpy as np
import cv2
import scipy.io
from numpy.linalg import inv
import math

def createNP():
    data_folder = "../MPIIFaceGaze"
    for i in range(0,10):
        anno_path = data_folder + "/p0{}/p0{}.txt".format(i)
        print(path)

        with open(anno_path, 'r') as ann:
            for num, line in enumerate(ann.readlines()):
                line_array = line.split(' ')

                headpose_hr = np.array(line_array[15:18], dtype="float64")
                headpose_ht = np.array(line_array[18:21], dtype="float64")

                face_center = np.array(line_array[21:24], dtype='float64')

                gaze_target = np.array(line_array[24:27], dtype='float64')

                cameraMatrix= scipy.io.loadmat(path)['cameraMatrix']

                image = cv2.imread(data_folder + "/p0{}/".format(i) + line_array[0], cv2.IMREAD_COLOR)
                if image is None:
                    print("the image path is wrong")
                norm_img, polar = normalize_data(faceModel, cameraMatrix, headpose_hr, headpose_ht, gaze_target, face_center, image)


    for i in range(0,5):
        path = data_folder + "/p1{}.mat".format(i)
        print(path)
        f = h5py.File(path, 'r')
        norm_pics = np.array(f.get('Data/data'))
        norm_pics = norm_pics.astype('uint8')
        norm_labs = np.array(f.get('Data/label'))
        np.save(path[:-4] + "_pics", norm_pics)
        np.save(path[:-4] + "_labs", norm_labs)


def normalize_data(faceModel, cameraMatrix, headpose_hr, headpose_ht, gaze_target, face_center, image):
    headpose_hR, trash = cv2.Rodrigues(headpose_hr)
    # moving facemodel into camera coordinates
    faceModel = np.matmul(headpose_hR, faceModel)
    faceModel = np.add(faceModel, np.array([headpose_ht]).T)

    norm_img, gaze, cnvMat = normalize_Image(image, face_center, headpose_hR, gaze_target, [488, 488], cameraMatrix)

    return norm_img, t(gaze / np.sqrt(gaze.dot(gaze)))


def normalize_Image(inputImg, target_3D, hR, gc, roiSize, cameraMatrix):
    focal_new = 960
    distance_new = 350
    distance = np.sqrt(target_3D.dot(target_3D))

    z_scale = distance_new / distance

    # camera matrix in normalized space
    cam_new = np.array([[focal_new, 0, roiSize[0]/2], [0, focal_new, roiSize[1]/2], [0, 0, 1]], dtype="float64")
    # matrix to scale the image
    scaleMat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, z_scale]], dtype="float64")
    #first row of rotation matrix of face
    # this describes the vector sticking out of the x coordinates axis after being rotated
    hRx = hR[:,0]
    #normalized target vector
    forward = target_3D / distance
    # taking an orthogonal unit vector to forward and the x part of rotation matrix
    down = np.cross(forward, hRx)
    down = down / np.sqrt(down.dot(down))
    # taking orthogonal unit vector orthogonal to  forward and down
    right = np.cross(down, forward)
    right = right / np.sqrt(right.dot(right))
    # did not transpose this matrix, even though the matlab code did
    rotMat = np.array([right, down, forward])

    warpMat = np.matmul( np.matmul(cam_new, scaleMat), np.matmul(rotMat, inv(cameraMatrix)) )

    img_warped = cv2.warpPerspective(inputImg, warpMat, tuple(roiSize))
    # normalizing gaze vector
    cnvMat = np.matmul(scaleMat, rotMat)
    gcnew = np.matmul(cnvMat, gc)
    htnew = np.matmul(cnvMat, target_3D)
    # the new gaze vector
    gvnew = np.subtract(gcnew, htnew)
    return img_warped, gvnew, cnvMat

def t(vec):
    theta = math.asin(-vec[1])
    phi = math.atan2(-vec[0], -vec[2])

    return [theta, phi]
