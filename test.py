import tensorflow as tf
import numpy as np
from medium_face import cnn_model_fn
import time
def main(unused_argv):
  # Load training and eval data

  data_folder = "../MPIIFaceGaze_kayl_norm"
  eval_data = np.load(data_folder + "/p00_data.npy").astype("float32") * (1.0/255.0)

  eval_labels = np.load(data_folder + "/p00_labels.npy").astype("float32")

  # eval_data = eval_data[100:101, :, :, :]
  # eval_labels = eval_labels[100:101,:]

  # Create the Estimator
  run_config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 0}))

  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="../model_medium", config=run_config)

  # Evaluate the model and print results
  # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
  #     x={"x": eval_data},
  #     y=eval_labels,
  #     num_epochs=1,
  #     shuffle=False)

  # eval_results = classifier.evaluate(input_fn=eval_input_fn)
  # print(eval_results)

  final = time.clock()
  for i in range(10):
    start = time.clock()
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data[i:i+1]},
        y=eval_labels[i:i+1],
        num_epochs=1,
        shuffle=False)
    pred_results = classifier.predict(input_fn=eval_input_fn, predict_keys=["angles"])
    print("time taken: ", time.clock() - start)
    print("eval labels: ", eval_labels[i:i+1])
    print("predictions: ", list(pred_results))
  print("total time taken: ", time.clock() - final)

if __name__ == "__main__":
    tf.app.run()
