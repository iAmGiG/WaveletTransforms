import tensorflow as tf
from absl import app
from absl import flags

FLAGS = flags.FLAGS


# Load the existing TensorFlow model
model = tf.keras.models.load_model(
    './DeepLearning/SavedStandardModels/mnist_model_02-20_17-53.h5')

# Set up the converter for post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_quant_model = converter.convert()

# Save the quantized model
with open('./DeepLearning/SavedStandardModels/mnist_model_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
    app.run(main)
