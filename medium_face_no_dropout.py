from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

model_directory = "../model_medium_no_dropout"
summary_directory = "../summary_medium_no_dropout"
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  with tf.variable_scope("Input"):
      input_layer = tf.reshape(features["x"], [-1, 448, 448, 3], name="Input")

  conv1_filters = 80
  conv2_filters = 128
  conv3_filters = 200
  dense_nodes = 3000
  # Convolutional Layer #1
  # outputs batchx112x112x96
  with tf.variable_scope("3_Layer_Convolution"):
      conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=conv1_filters,
          kernel_size=[4, 4],
          strides=[4,4],
          padding="valid",
          activation=tf.nn.relu,
          name="Convolution_1")

      # Pooling Layer #1
      # outputs batchx56x56x96
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="Pool_1")

      # Convolutional Layer #2
      # outputs
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=conv2_filters,
          kernel_size=[2, 2],
          padding="same",
          activation=tf.nn.relu,
          name="Convolution_2")

      # Pooling Layer #2
      #outputs batchx28X28X128
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="Pool_2")

      # outputs batchx28x28x256
      conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=conv3_filters,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu,
        name="Convolution_3"
      )

      # outputs batchx14x14x256
      pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2, name="Pool_3")

  heat_conv1 = tf.layers.conv2d(
  inputs=pool3,
  filters=conv3_filters,
  kernel_size=[1,1],
  padding="same",
  activation=tf.nn.relu,
  name="Heat_Conv_1"
  )
  heat_conv2 = tf.layers.conv2d(
  inputs=heat_conv1,
  filters=conv3_filters,
  kernel_size=[1,1],
  padding="same",
  activation=tf.nn.relu,
  name="Heat_Conv_2"
  )
  # batchx14x14x1
  heat_map = tf.layers.conv2d(
  inputs=heat_conv2,
  filters=1,
  kernel_size=[1,1],
  padding="same",
  activation=tf.nn.relu,
  name="Heat_Map"
  )
  with tf.variable_scope("Reshaping"):
      flat_heat_map = tf.reshape(heat_map, [-1, 14*14])
      #repeats the heat map 256 times
      flat_heat_map_resized = tf.tile(flat_heat_map, [1, conv3_filters])

  with tf.variable_scope("Reshaping"):
      # size batchx14*14*256
      flat_pool3 = tf.reshape(pool3, [-1, 14*14*conv3_filters])

  #because of numpy broadcasting will do component wise multiplication
  #batchx14*14*256 times batchx14*14*256
  flat_weighted_pool3 = tf.multiply(flat_pool3, flat_heat_map_resized)

  with tf.variable_scope("Dense_Layer"):
      dense1 = tf.layers.dense(inputs=flat_weighted_pool3, units=dense_nodes, activation=tf.nn.relu, name="Dense_1")


      dense2 = tf.layers.dense(inputs=dense1, units=dense_nodes, activation=tf.nn.relu, name="Dense_2")

  # prediction layer
  prediction = tf.layers.dense(inputs=dense2, units=2, name="Prediction")

  #CHANGE PREDICTIONS TO NOT BE
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "angles": prediction,
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.absolute_difference(labels, prediction)

  writer = tf.summary.FileWriter(summary_directory, tf.get_default_graph())
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "mean absolute error": tf.metrics.mean_absolute_error(
          labels=labels, predictions=predictions["angles"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=model_directory)

  # Load training and eval data
  data_folder = "../MPIIFaceGaze_kayl_norm"

  epoch_count = 100
  for x in range(0,epoch_count):
      print(" ")
      print("Epoch number: ", x)
      print(" ")
      for i in range(0,15):
          if i < 10:
              pathp = data_folder + "/p0{}_data.npy".format(i)
              pathl = data_folder + "/p0{}_labels.npy".format(i)
              print(pathp)
              train_data = np.load(pathp).astype("float32") * (1.0/255.0)
              #train_data = train_data.astype("float32") * (1.0/255.0)
              train_labels = np.load(pathl).astype("float32")
              print("train label dtype: {}, train data dtype: {}".format(train_labels.dtype, train_data.dtype))
              train_input_fn = tf.estimator.inputs.numpy_input_fn(
                  x={"x": train_data},
                  y=train_labels,
                  batch_size=128,
                  num_epochs=1,
                  shuffle=True)
              classifier.train(
                input_fn=train_input_fn,
                steps=None,
                hooks=[])#took out logging_hook
          else:
              pathp = data_folder + "/p1{}_data.npy".format(i-10)
              pathl = data_folder + "/p1{}_labels.npy".format(i-10)
              print(pathp)
              train_data = np.load(pathp).astype("float32") * (1.0/255.0)
              train_labels = np.load(pathl).astype("float32")

              train_input_fn = tf.estimator.inputs.numpy_input_fn(
                  x={"x": train_data},
                  y=train_labels,
                  batch_size=128,
                  num_epochs=1,
                  shuffle=True)

              classifier.train(
                input_fn=train_input_fn,
                steps=None,
                hooks=[])#took out logging_hook


if __name__ == "__main__":
  tf.app.run()
