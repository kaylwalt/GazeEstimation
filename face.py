from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 448, 448, 3])

  # Convolutional Layer #1
  # outputs batchx112x112x96
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[4, 4],
      strides=[4,4],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # outputs batchx56x56x96
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # outputs
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  #outputs batchx28X28X128
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # outputs batchx28x28x256
  conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=256,
    kernel_size=[2,2],
    padding="same",
    activation=tf.nn.relu
  )

  # outputs batchx14x14x256
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

  heat_conv1 = tf.layers.conv2d(
  inputs=pool3,
  filters=256,
  kernel_size=[1,1],
  padding="same",
  activation=tf.nn.relu
  )
  heat_conv2 = tf.layers.conv2d(
  inputs=heat_conv1,
  filters=256,
  kernel_size=[1,1],
  padding="same",
  activation=tf.nn.relu
  )
  # batchx14x14x1
  heat_map = tf.layers.conv2d(
  inputs=heat_conv1,
  filters=1,
  kernel_size=[1,1],
  padding="same",
  activation=tf.nn.relu
  )

  flat_heat_map = tf.reshape(heat_map, [-1, 14*14])
  #repeats the heat map 256 times
  flat_heat_map_resized = tf.tile(flat_heat_map, [1, 256])

  # size batchx14*14*256
  flat_pool3 = tf.reshape(pool3, [-1, 14*14*256])

  #because of numpy broadcasting will do component wise multiplication
  #batchx14*14*256 times batchx14*14*256
  flat_weighted_pool3 = tf.multiply(flat_pool3, flat_heat_map_resized)

  dense1 = tf.layers.dense(inputs=flat_weighted_pool3, units=4096, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  dense2 = tf.layers.dense(inputs=dropout, units=4096, activation=tf.nn.relu)

  # prediction layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  prediction = tf.layers.dense(inputs=dense2, units=2)

  #CHANGE PREDICTIONS TO NOT BE
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "angles": prediction,
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.absolute_difference(labels, prediction)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "mean squared error": tf.metrics.mean_squared_error(
          labels=labels, predictions=predictions["angles"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #logging_hook = tf.train.LoggingTensorHook(
  #   tensors=tensors_to_log, every_n_iter=50)

  # Load training and eval data
  data_folder = "../MPIIFaceGaze_normalized"
  train_data = np.load(data_folder + "/p00_pics.npy")
  train_data = np.swapaxes(np.swapaxes(train_data, 1, 3), 1, 2)
  train_data = train_data.astype("float32")
  train_labels = np.load(data_folder + "/p00_labs.npy")
  # Train the model

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=64,
      num_epochs=1,
      shuffle=True)

  classifier.train(
    input_fn=train_input_fn,
    steps=None,
    hooks=[])#took out logging_hook
  for i in range(12,15):
      if i < 10:
          pathp = data_folder + "/p0{}_pics.npy".format(i)
          pathl = data_folder + "/p0{}_labs.npy".format(i)
          print(pathp)
          train_data = np.load(pathp)
          train_data = np.swapaxes(np.swapaxes(train_data, 1, 3), 1, 2)
          train_data = train_data.astype("float32")
          train_labels = np.load(pathl)
          classifier.train(
            input_fn=train_input_fn,
            steps=None,
            hooks=[])#took out logging_hook
      else:
          pathp = data_folder + "/p1{}_pics.npy".format(i-10)
          pathl = data_folder + "/p1{}_labs.npy".format(i-10)
          print(pathp)
          train_data = np.load(pathp)
          train_data = np.swapaxes(np.swapaxes(train_data, 1, 3), 1, 2)
          train_data = train_data.astype("float32")
          train_labels = np.load(pathl)
          classifier.train(
            input_fn=train_input_fn,
            steps=None,
            hooks=[])#took out logging_hook


if __name__ == "__main__":
  tf.app.run()
