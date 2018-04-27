#This is original and ran on the data 3 times, with 2 people chosen to be left out, the network was retrained each time
#after randomly renaming the participant numpy data arrays, and then tested with this program
#
import tensorflow as tf
import numpy as np
from medium_face import cnn_model_fn
import time
from helper_fun import angle_dist, trev

def main(unused_argv):
  # Load training and eval data
  run_config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 1}))
  print("making the classifier")
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="../model_medium_leave_2", config=run_config)


  data_folder = "../MPIIFaceGaze_kayl_norm_leave_one"
  print("Loading the data")
  eval_data = np.load(data_folder + "/p00_data.npy").astype("float32") * (1.0/255.0)
  eval_data2 = np.load(data_folder + "/p01_data.npy").astype("float32") * (1.0/255.0)

  eval_labels = np.load(data_folder + "/p00_labels.npy").astype("float32")
  eval_labels2 = np.load(data_folder + "/p01_labels.npy").astype("float32")

  test_data = np.concatenate((eval_data, eval_data2))
  test_labels =np.concatenate((eval_labels, eval_labels2))

  pred_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      num_epochs=1,
      shuffle=False)
  pred_results = classifier.predict(input_fn=pred_input_fn)
  pred_act = list(zip(list(map(lambda x: x['angles'], pred_results)), test_labels))
  errors = np.rad2deg(list(map(lambda x : angle_dist(np.squeeze(trev(x[0])), np.squeeze(trev(x[1]))), pred_act)))
  print("average error in degrees: ", np.mean(errors))


if __name__ == "__main__":
    tf.app.run()
