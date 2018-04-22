import cv2
import yaml
from imutils import face_utils
import numpy as np
import imutils
import dlib
import time
from normalize_dataset_my_data import normalize_data

with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
camera_matrix = np.array(loadeddict.get('camera_matrix'))
dist_coeff = np.array(loadeddict.get('dist_coeff'))
face_model = np.load("face_model.npy")

print("camera matrix: ", camera_matrix)
print("Dis coeff: ", dist_coeff)
print("face model: ", face_model.T)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
#set height
cap.set(3, 1080)
#set width
cap.set(4, 1920)
ret, frame = cap.read()

# shave_row = 5.0
# shave_col = 5.0
#
# lower_row_cut = int(frame.shape[0] / shave_row)
# upper_row_cut = int(frame.shape[0] * ((shave_row - 1)/shave_row))
# lower_col_cut = int(frame.shape[1] / shave_col)
# upper_col_cut = int(frame.shape[1] * ((shave_col - 1)/shave_col))
# frame = frame[ lower_row_cut: upper_row_cut,
#                lower_col_cut: upper_col_cut]

gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

start = time.clock()
rects = detector(gray, 1)

if len(rects) == 0:
    print("no rects")



font = cv2.FONT_HERSHEY_SIMPLEX

points = []
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    for i, x in enumerate([36, 39, 42, 45, 48, 54]):
        points.append(tuple(shape[x]))
        cv2.circle(frame, tuple(shape[x]), 3, (0,255,0), -1)
        cv2.putText(frame, str(i), tuple([shape[x][0] + 3, shape[x][1] + 3]), font, 0.8, (0, 255, 0))

points = np.array(points, dtype="float32")

print(face_model.T)
print(points)
retval, rvec, tvec = cv2.solvePnP(face_model.T, points, camera_matrix, dist_coeff)

print("rvec and tvec")
rvec = np.squeeze(rvec)
tvec =  np.squeeze(tvec)
print(rvec)
print(tvec)
print("Time to detect: ", time.clock() - start)

gaze_t = np.array([0, 0, -100])


normalized_image, gaze_vec, cnvMat = normalize_data(face_model, camera_matrix, rvec, tvec.T, gaze_t,
                                    None, frame)

while(True):
    cv2.imshow('before', frame)
    cv2.imshow('img1', normalized_image)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('c1.png', frame)
        cv2.destroyAllWindows()
        break
cap.release()
