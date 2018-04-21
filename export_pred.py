import tensorflow as tf
from medium_face import cnn_model_fn

model_directory = "../model_medium"

def serving_input_receiver_fn():
  """Build the serving inputs."""
  # The outer dimension (None) allows us to batch up inputs for
  # efficiency. However, it also means that if we want a prediction
  # for a single instance, we'll need to wrap it in an outer list.
  inputs = {"x": tf.placeholder(shape=[None, 448, 448, 3], dtype=tf.float32)}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# Create the Estimator
classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn, model_dir=model_directory)

export_dir = classifier.export_savedmodel(
    export_dir_base="../export_model_medium",
    serving_input_receiver_fn=serving_input_receiver_fn)

print(export_dir)
