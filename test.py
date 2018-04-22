import tensorflow as tf
import numpy as np
from medium_face import cnn_model_fn
import time



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

def main(unused_argv):
  # Load training and eval data

  data_folder = "../MPIIFaceGaze_kayl_norm"
  eval_data = np.load(data_folder + "/p00_data.npy").astype("float32") * (1.0/255.0)

  eval_labels = np.load(data_folder + "/p00_labels.npy").astype("float32")

  # eval_data = eval_data[100:101, :, :, :]
  # eval_labels = eval_labels[100:101,:]

  # Create the Estimator
  run_config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 1}))

  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="../model_medium", config=run_config)

  fp = FastPredict(classifier)
  # Evaluate the model and print results
  # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
  #     x={"x": eval_data},
  #     y=eval_labels,
  #     num_epochs=1,
  #     shuffle=False)

  # eval_results = classifier.evaluate(input_fn=eval_input_fn)
  # print(eval_results)

  final = time.clock()
  num_entries = 1000
  for i in range(num_entries):
    start = time.clock()
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data[i:i+1]},
        num_epochs=1,
        shuffle=False)
    #print(tuple(eval_data[i:i+1]))
    #pred_results = classifier.predict(input_fn=pred_input_fn)
    pred_results = fp.predict({"x": eval_data[i:i+1]})
    #print("time taken for {}: ".format(i), time.clock() - start)
    #print("eval labels: ", eval_labels[i:i+1])
    #print("predictions: ", list(pred_results))
  total_time = time.clock() - final
  print("total time taken: ", total_time)
  print("predictions per second: ", num_entries / total_time)

if __name__ == "__main__":
    tf.app.run()
