#landmark detection implementation from [1], https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
#FastPredict Based off of code from [2], https://github.com/marcsto/rl/blob/master/src/fast_predict.py, just modified for
#the new tensorflow api, which takes a Dataset.from_generator instead of a straight python generator
#
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
from medium_face_no_dropout import cnn_model_fn
from numpy.linalg import inv

#Start of [2]
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

#end of [2]

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

    #start of code based of [1]
    gray = cv2.cvtColor(frame_cut, cv2.COLOR_RGB2GRAY)
    rects = face_detector(gray, 1)

    if len(rects) == 0:
        print("no rects")
        raise ValueError("No face was found in the picture")


    points = []
    for (i, rect) in enumerate(rects):
        shape = face_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for i, x in enumerate([36, 39, 42, 45, 48, 54]):
            points.append(tuple([shape[x][0] + lower_col_cut, shape[x][1] + lower_row_cut]))


    points = np.array(points, dtype="float32")
    #end of code based off [1]

    retval, rvec, tvec = cv2.solvePnP(face_model.T, points, camera_matrix, dist_coeff)


    rvec = np.squeeze(rvec)
    tvec =  np.squeeze(tvec)

    gaze_t = np.array([0, 0, -1])
    normalized_image, gaze_vec, cnvMat, face_center = normalize_data(face_model, camera_matrix, rvec, tvec.T, gaze_t,
                                        None, frame)

    d = np.array([normalized_image * (1.0/255.0)], dtype="float32")

    predictions = fast_predictor.predict({"x": d})

    if __name__ == "__main__":
        while(True):
            cv2.imshow('before', frame)
            cv2.imshow('img1', normalized_image)
            if cv2.waitKey(1) & 0xFF == ord('y'):
                cv2.imwrite('c1.png', frame)
                cv2.destroyAllWindows()
                break

    #returning a vector going from the center of the face to the unit vector translated from the
    #normalized image space back to normal camera space
    vec = np.array(trev(list(predictions[0]['angles'])))
    cam_space_vec = np.squeeze(np.matmul(inv(cnvMat), vec))
    cam_space_vec = cam_space_vec / np.sqrt(cam_space_vec.dot(cam_space_vec))
    return cam_space_vec, face_center


def main():
    model_dir = "../model_medium_no_dropout_short_train/"

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
