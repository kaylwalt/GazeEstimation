import tensorflow as tf
import numpy as np
from face import cnn_model_fn
def main(unused_argv):
  # Load training and eval data

  data_folder = "../MPIIFaceGaze_normalized"
  eval_data = np.load(data_folder + "/p00_pics.npy")
  eval_data = np.swapaxes(np.swapaxes(eval_data, 1, 3), 1, 2)
  eval_data = eval_data.astype("float32")
  eval_labels = np.load(data_folder + "/p00_labs.npy")
  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./model")

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
    tf.app.run()
