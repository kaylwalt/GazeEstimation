import os
import numpy as np
import cv2
import scipy.io
from numpy.linalg import inv
import math

def normalize_data_set():
    data_folder = "../MPIIFaceGaze"
    norm_folder = "../MPIIFaceGaze_kayl_norm"
    if os.path.isdir(norm_folder):
        print("using norm folder")
    else:
        print("making folder: ", norm_folder)
        os.makedirs(norm_folder)

    faceModel = np.load("face_model.npy")
    for i in range(0,10):
        anno_path = data_folder + "/p0{}/p0{}.txt".format(i, i)
        print(anno_path)
        data = []
        labels = []
        with open(anno_path, 'r') as ann:
            for num, line in enumerate(ann.readlines()):
                line_array = line.split(' ')

                headpose_hr = np.array(line_array[15:18], dtype="float64")
                headpose_ht = np.array(line_array[18:21], dtype="float64")

                face_center = np.array(line_array[21:24], dtype='float64')

                gaze_target = np.array(line_array[24:27], dtype='float64')

                cameraMatrix= scipy.io.loadmat(data_folder + "/p0{}/Calibration/Camera.mat".format(i))['cameraMatrix']

                image = cv2.imread(data_folder + "/p0{}/".format(i) + line_array[0], cv2.IMREAD_COLOR)
                if image is None:
                    print("the image path is wrong")
                norm_img, polar = normalize_data(faceModel, cameraMatrix, headpose_hr, headpose_ht, gaze_target, face_center, image)
                data.append(norm_img)
                labels.append(polar)
            np_data = np.array(data, dtype="uint8")
            np_labels = np.array(labels, dtype="float64")
            print("data shape: ", np_data.shape)
            print("label shape: ", np_labels.shape)
            np.save(norm_folder + "/p0{}_data.npy".format(i), np_data)
            np.save(norm_folder + "/p0{}_labels.npy".format(i), np_labels)

    for i in range(0,5):
        anno_path = data_folder + "/p1{}/p1{}.txt".format(i, i)
        print(anno_path)
        data = []
        labels = []
        with open(anno_path, 'r') as ann:
            for num, line in enumerate(ann.readlines()):
                line_array = line.split(' ')

                headpose_hr = np.array(line_array[15:18], dtype="float64")
                headpose_ht = np.array(line_array[18:21], dtype="float64")

                face_center = np.array(line_array[21:24], dtype='float64')

                gaze_target = np.array(line_array[24:27], dtype='float64')

                cameraMatrix= scipy.io.loadmat(data_folder + "/p1{}/Calibration/Camera.mat".format(i))['cameraMatrix']

                image = cv2.imread(data_folder + "/p1{}/".format(i) + line_array[0], cv2.IMREAD_COLOR)
                if image is None:
                    print("the image path is wrong")
                norm_img, polar = normalize_data(faceModel, cameraMatrix, headpose_hr, headpose_ht, gaze_target, face_center, image)
                data.append(norm_img)
                labels.append(polar)

            np_data = np.array(data, dtype="uint8")
            np_labels = np.array(labels, dtype="float64")
            print("data shape: ", np_data.shape)
            print("label shape: ", np_labels.shape)
            np.save(norm_folder + "/p1{}_data.npy".format(i), np_data)
            np.save(norm_folder + "/p1{}_labels.npy".format(i), np_labels)



def normalize_data(faceModel, cameraMatrix, headpose_hr, headpose_ht, gaze_target, face_center, image):
    headpose_hR, trash = cv2.Rodrigues(headpose_hr)
    headpose_hR = headpose_hR
    # moving facemodel into camera coordinates
    faceModel = np.matmul(headpose_hR, faceModel)
    faceModel = np.add(faceModel, np.array([headpose_ht]).T)

    if face_center is None:
        #print(faceModel)
        #print(faceModel.shape)
        face_center = faceModel.mean(axis=1)
        #print("face center: ", face_center)

    norm_img, gaze, cnvMat = normalize_Image(image, face_center, headpose_hR, gaze_target, [448, 448], cameraMatrix)

    return norm_img, t(gaze / np.sqrt(gaze.dot(gaze))), cnvMat, face_center


def normalize_Image(inputImg, target_3D, hR, gc, roiSize, cameraMatrix):
    focal_new = 960
    distance_new = 350
    distance = np.sqrt(target_3D.dot(target_3D))

    z_scale = distance_new / distance

    # camera matrix in normalized space
    cam_new = np.array([[focal_new, 0, roiSize[0]/2], [0, focal_new, roiSize[1]/2], [0, 0, 1]], dtype="float32")
    # matrix to scale the image
    scaleMat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, z_scale]], dtype="float32")
    #first row of rotation matrix of face
    # this describes the vector sticking out of the x coordinates axis after being rotated
    hRx = hR[:,0] * -1
    #normalized target vector
    forward = (target_3D / distance)
    # taking an orthogonal unit vector to forward and the x part of rotation matrix
    down = np.cross(forward, hRx)
    down = down / np.sqrt(down.dot(down))
    # taking orthogonal unit vector orthogonal to  forward and down
    right = np.cross(down, forward)
    right = right / np.sqrt(right.dot(right)) * -1
    # did not transpose this matrix, even though the matlab code did
    rotMat = np.array([right, down, forward])

    warpMat = np.matmul( np.matmul(cam_new, scaleMat), np.matmul(rotMat, inv(cameraMatrix)) )

    img_warped = cv2.warpPerspective(inputImg, warpMat, tuple(roiSize))

    img_yuv = cv2.cvtColor(img_warped, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_warped = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

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

if __name__ == "__main__":
    normalize_data_set()
