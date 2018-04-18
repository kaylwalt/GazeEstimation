import numpy as np
import cv2
import scipy.io
from numpy.linalg import inv
import math
import sys
import time

faceModel = np.load("face_model.npy")
print("-------faceModel------")
print(faceModel)

print(sys.argv)
with open("../MPIIFaceGaze/p00/p00.txt", 'r') as ann:
    for num, line in enumerate(ann.readlines()):
        if num != int(sys.argv[1]):
            continue

        line_array = line.split(' ')
        print("-------annotations-------")
        for i, item in enumerate(line_array):
            print("index ", i, ": dimension ", i + 1, ": ", item)

        # headpose rotation and translation vectors to convert between
        # face coordinates and camera coordinates
        headpose_hr = np.array(line_array[15:18], dtype="float64")
        headpose_ht = np.array(line_array[18:21], dtype="float64")
        print("-------headpose_hr and headpose_ht-------")
        print(headpose_hr)
        print(headpose_ht)
        #gaze target in the camera coordinate system
        face_center = np.array(line_array[21:24], dtype='float64')
        print("-------face_center-------")
        print(face_center)

        gaze_target = np.array(line_array[24:27], dtype='float64')
        print("-------gaze_target-------")
        print(gaze_target)

        cameraMatrix= scipy.io.loadmat("../MPIIFaceGaze/p00/Calibration/Camera.mat")['cameraMatrix']
        print("-------cameraMatrix-------")
        print(cameraMatrix)

        image = cv2.imread("../MPIIFaceGaze/p00/" + line_array[0], cv2.IMREAD_COLOR)
        if image is None:
            print("the image path is wrong")



def normalize_data(faceModel, cameraMatrix, headpose_hr, headpose_ht, gaze_target, face_center, image):
    headpose_hR, trash = cv2.Rodrigues(headpose_hr)
    print("-------headpose_hR-------")
    print(headpose_hR)
    # moving facemodel into camera coordinates
    faceModel = np.matmul(headpose_hR, faceModel)
    faceModel = np.add(faceModel, np.array([headpose_ht]).T)
    print("--------new face model----------")
    print(faceModel)
    print("calculated center of face: ", faceModel.mean(axis=1))
    print("read out center of face: ", face_center)
    cv2.imshow("image", image)
    # cv2.waitKey(0)
    norm_img, gaze, cnvMat = normalize_Image(image, face_center, headpose_hR, gaze_target, [448, 448], cameraMatrix)
    print("original gaze vector: ", np.subtract(gaze_target, face_center))
    print("reverted gaze vector: ", np.matmul(inv(cnvMat), gaze))
    oggazevec = np.subtract(gaze_target, face_center)
    oggazevec = oggazevec / np.sqrt(oggazevec.dot(oggazevec))
    print("normalized gaze vector: ", oggazevec)
    print("Theta phi represenation: ", t(oggazevec))
    print("normalized image size: ", norm_img.shape)
    print("original image size: ", image.shape)
    cv2.imshow("normalized image", norm_img)

    cv2.waitKey(0)

    return norm_img


def normalize_Image(inputImg, target_3D, hR, gc, roiSize, cameraMatrix):
    focal_new = 960
    distance_new = 350
    distance = np.sqrt(target_3D.dot(target_3D))
    print("distance: ", distance)
    z_scale = distance_new / distance
    print("z scale: ", z_scale)
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
    print("---------forward------------")
    print(forward)
    print("---------down-------")
    print(down)
    print("----------right-----------")
    print(right)
    # did not transpose this matrix, even though the matlab code did
    rotMat = np.array([right, down, forward])
    print("-------rotmat----------")
    print(rotMat)

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
    start = time.clock()
    normalize_data(faceModel, cameraMatrix, headpose_hr, headpose_ht, gaze_target, face_center, image)
    print("time to normalize image: ", time.clock() - start)
