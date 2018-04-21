from tensorflow.contrib import predictor
import numpy as np
import sys
import tensorflow as tf
import math

def trev(angles):
    """input as [theta, phi], [rot around x, rot around y]"""
    angles = [-angles[0], -angles[1]]
    rot_mat_x = np.array([[1, 0, 0],
                          [0, math.cos(angles[0]), -math.sin(angles[0])],
                          [0, math.sin(angles[0]), math.cos(angles[0])]])
    rot_mat_y = np.array([[math.cos(angles[1]), 0, -math.sin(angles[1])],
                          [0, 1, 0],
                          [math.sin(angles[1]), 0, math.cos(angles[1])]])
    start = np.array([[0], [0], [-1]])

    final_rot = np.matmul(rot_mat_x, rot_mat_y)
    answer = np.matmul(final_rot, start)

    return answer

def angle_dist(u1, u2):
    return np.arccos(u1.dot(u2))


export_dir = "../export_model_medium/1524331786"

data_folder = "../MPIIFaceGaze_kayl_norm"
eval_data = np.load(data_folder + "/p05_data.npy").astype("float32") * (1.0/255.0)

eval_labels = np.load(data_folder + "/p05_labels.npy").astype("float32")

index = int(sys.argv[1])
num_in = 200

predict_fn = predictor.from_saved_model(export_dir)
predictions = predict_fn(
    {"x": eval_data[index:index+num_in]})

labels = eval_labels[index:index+num_in]
#print("predicted: ", predictions['output'])
#print("real values: ", labels)

dif = np.abs(predictions['output'] - labels)
print("mean absolute difference", np.mean(dif))

lab_vecs = list(map(trev, labels))
pred_vecs = list(map(trev, predictions['output']))

pairs = list(zip(lab_vecs, pred_vecs))
#print(pairs)
errors = list(map(lambda x: angle_dist(np.squeeze(np.array(x[0])), np.squeeze(np.array(x[1]))), pairs))
#print("angle errors:", errors)
print("average angle error degrees: ", math.degrees(np.mean(errors)))
print("average angle error radians: ", np.mean(errors))
