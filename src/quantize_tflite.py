import numpy as np
import tensorflow as tf

# We give it the observation space which should include the maximum range
# of values.
# Generally simulation data only includes a small range of data, and when
# we use the model in unexpected ways may get values outside the quantized
# range.
dataset = [
  [-3.14 / 2, -6.28, -4, -4, -4, -4],
  [0, 0, 0, 0, 0, 0],
  [3.14 / 2, 6.28, 4, 4, 4, 4],
]

def representative_dataset():
  for data in dataset:
    data_scaled = [d * 1.0 for d in data]
    yield {
      "input": np.array(data_scaled, dtype=np.float32),
    }

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
  tf.lite.OpsSet.TFLITE_BUILTINS   # TODO: possibly remove
]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
with open('saved_model/int8_model.tflite', 'wb') as w:
  w.write(tflite_quant_model)


