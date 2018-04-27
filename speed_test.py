#Based off of code from [1], https://github.com/marcsto/rl/blob/master/src/fast_predict.py, just modified for
#the new tensorflow api, which takes a Dataset.from_generator instead of a straight python generator
#
import tensorflow as tf
import numpy as np
from medium_face import cnn_model_fn
import time

# start of code from [1]
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

#end of code from [1]

def main(unused_argv):
  # Load training and eval data

  data_folder = "../MPIIFaceGaze_kayl_norm"
  print("Loading the data")
  eval_data = np.load(data_folder + "/p00_data.npy").astype("float32") * (1.0/255.0)

  eval_labels = np.load(data_folder + "/p00_labels.npy").astype("float32")

  # Create the Estimator
  run_config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 1}))
  print("making the classifier")
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="../model_medium_leave_2", config=run_config)

  fp = FastPredict(classifier)


  final = time.clock()
  num_entries = 10
  for i in range(num_entries):
    start = time.clock()
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data[i:i+1]},
        num_epochs=1,
        shuffle=False)

    pred_results = fp.predict({"x": eval_data[i:i+1]})
    print("eval labels: ", eval_labels[i:i+1])
    print("predictions: ", list(pred_results))
  total_time = time.clock() - final
  print("total time taken: ", total_time)
  print("predictions per second: ", num_entries / total_time)

if __name__ == "__main__":
    tf.app.run()
