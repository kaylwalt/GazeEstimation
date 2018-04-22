import cv2
import yaml
from imutils import face_utils
import numpy as np
import imutils
import dlib
import time
from normalize_dataset_my_data import normalize_data
import tensorflow as tf
from tensorflow.contrib import predictor
from helper_fun import trev, angle_dist

def predict_gaze(gaze_predict_fn, cap, face_detector, face_predictor, camera_matrix, dist_coeff, face_model):
    start = time.clock()
    ret, frame = cap.read()

    shave_row = 5.0
    shave_col = 5.0

    lower_row_cut = int(frame.shape[0] / shave_row)
    upper_row_cut = int(frame.shape[0] * ((shave_row - 1)/shave_row))
    lower_col_cut = int(frame.shape[1] / shave_col)
    upper_col_cut = int(frame.shape[1] * ((shave_col - 1)/shave_col))
    frame_cut = frame[ lower_row_cut: upper_row_cut,
                   lower_col_cut: upper_col_cut]

    gray = cv2.cvtColor(frame_cut, cv2.COLOR_RGB2GRAY)
    rects = face_detector(gray, 1)

    if len(rects) == 0:
        print("no rects")

    #font = cv2.FONT_HERSHEY_SIMPLEX

    points = []
    for (i, rect) in enumerate(rects):
        shape = face_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for i, x in enumerate([36, 39, 42, 45, 48, 54]):
            points.append(tuple([shape[x][0] + lower_col_cut, shape[x][1] + lower_row_cut]))
            # cv2.circle(frame, tuple([shape[x][0] + lower_col_cut, shape[x][1] + lower_row_cut]), 3, (0,255,0), -1)
            # cv2.putText(frame, str(i), tuple([shape[x][0] + 3, shape[x][1] + 3]), font, 0.8, (0, 255, 0))

    points = np.array(points, dtype="float32")

    # print(face_model.T)
    # print(points)
    retval, rvec, tvec = cv2.solvePnP(face_model.T, points, camera_matrix, dist_coeff)

    # print("rvec and tvec")
    rvec = np.squeeze(rvec)
    tvec =  np.squeeze(tvec)
    # print(rvec)
    # print(tvec)

    gaze_t = np.array([0, 0, -1])


    normalized_image, gaze_vec, cnvMat = normalize_data(face_model, camera_matrix, rvec, tvec.T, gaze_t,
                                        None, frame)
    # print("normalized image dtype: ", normalized_image.dtype)
    predictions = gaze_predict_fn(
        {"x": np.array([normalized_image * (1.0/255.0)], dtype="float32")})
    print("Time to predict: ", time.clock() - start)
    print("predictions in radians: ", list(predictions['output']))
    print("predictions in degrees: ", list(np.rad2deg(predictions['output'])))

    while(True):
        cv2.imshow('before', frame)
        cv2.imshow('img1', normalized_image)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            cv2.imwrite('c1.png', frame)
            cv2.destroyAllWindows()
            break


def main():
    export_dir = "../export_model_medium/1524331786"
    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f)
    camera_matrix = np.array(loadeddict.get('camera_matrix'))
    dist_coeff = np.array(loadeddict.get('dist_coeff'))
    face_model = np.load("face_model.npy")

    print("camera matrix: ", camera_matrix)
    print("Dis coeff: ", dist_coeff)
    print("face model: ", face_model.T)

    detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)
    #set height
    cap.set(3, 1080)
    #set width
    cap.set(4, 1920)
    predict_fn = predictor.from_saved_model(export_dir)
    predict_gaze(predict_fn, cap, detector, dlib_predictor, camera_matrix, dist_coeff, face_model)
    cap.release()
if __name__ == "__main__":
    main()
