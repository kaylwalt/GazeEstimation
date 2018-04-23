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
from helper_fun import trev, angle_dist, LinePlaneCollision
from medium_face import cnn_model_fn
from numpy.linalg import inv

class FastPredict:
    def _createGenerator(self):
        while not self.closed:
            yield self.next_features

    def __init__(self, estimator):
        self.estimator = estimator
        self.first_run = True
        self.closed = False


    def predict(self, features):
        self.next_features = features
        if self.first_run:
            def data_input_fn():
                ds = tf.data.Dataset.from_generator(self._createGenerator,
                                        {"x": tf.float32})
                return ds
            self.predictions = self.estimator.predict(data_input_fn)
            self.first_run = False

        results = []
        results.append(next(self.predictions))
        return results

    def close(self):
        self.closed=True
        next(self.predictions)


def predict_gaze(fast_predictor, cap, face_detector, face_predictor, camera_matrix, dist_coeff, face_model):
    #start = time.clock()
    ret, frame = cap.read()

    shave_row = 8.0
    shave_col = 8.0

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
        raise ValueError("No face was found in the picture")
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
    normalized_image, gaze_vec, cnvMat, face_center = normalize_data(face_model, camera_matrix, rvec, tvec.T, gaze_t,
                                        None, frame)

    d = np.array([normalized_image * (1.0/255.0)], dtype="float32")
    #intime = time.clock()
    predictions = fast_predictor.predict({"x": d})
    # print("tf time: ", time.clock() - intime)
    # print("Time to predict: ", time.clock() - start)
    # print("predictions in radians: ", predictions[0]['angles'])
    # print("predictions in degrees: ", list(np.rad2deg(predictions[0]['angles'])))
    #
    # while(True):
    #     cv2.imshow('before', frame)
    #     cv2.imshow('img1', normalized_image)
    #     if cv2.waitKey(1) & 0xFF == ord('y'):
    #         cv2.imwrite('c1.png', frame)
    #         cv2.destroyAllWindows()
    #         break

    #returning a vector going from the center of the face to the unit vector translated from the
    #normalized image space back to normal camera space
    vec = np.array(trev(list(predictions[0]['angles'])))
    cam_space_vec = np.squeeze(np.matmul(inv(cnvMat), vec))
    cam_space_vec = cam_space_vec / np.sqrt(cam_space_vec.dot(cam_space_vec))
    return cam_space_vec, face_center


def main():
    model_dir = "../model_medium/"
    # Create the Estimator
    # run_config = tf.estimator.RunConfig().replace(
    #   session_config=tf.ConfigProto(device_count={'GPU': 0}))
    print("making the classifier")
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)
    fast = FastPredict(classifier)
    print("initializing the graph")
    trash = fast.predict({"x": np.zeros((1,448,448,3), dtype="float32")})

    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f)
    camera_matrix = np.array(loadeddict.get('camera_matrix'))
    dist_coeff = np.array(loadeddict.get('dist_coeff'))
    face_model = np.load("face_model.npy")

    #print("camera matrix: ", camera_matrix)
    #print("Dis coeff: ", dist_coeff)
    #print("face model: ", face_model.T)

    detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)
    #set height
    cap.set(3, 1080)
    #set width
    cap.set(4, 1920)

    #vec is the vector in camera space pointing from face center to the gaze point
    cam_dir_vec, face_Center = predict_gaze(fast, cap, detector, dlib_predictor, camera_matrix, dist_coeff, face_model)
    print("direction: ", cam_dir_vec)
    print("face center: ", face_Center)
    p = LinePlaneCollision(np.array([0, 0, 1]), np.array([0,0,0]), cam_dir_vec, face_Center)
    print(p)
    cap.release()
if __name__ == "__main__":
    main()
