import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST test dataset
(testX, testY), (_, _) = mnist.load_data()
testX = testX.astype('float32') / 255.0
testX = np.expand_dims(testX, -1)  # Make sure to match the model's input shape
testY = tf.keras.utils.to_categorical(testY)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(
    model_path='./DeepLearning/SavedStandardModels/mnist_model_nn_quant.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to evaluate accuracy


def evaluate_model(interpreter, testX, testY):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    
    prediction_digits = []
    for test_image in testX:
        # Ensure test_image is float32
        test_image = test_image.astype(np.float32)
        
        # Make sure test_image is of shape [28, 28] (or [28, 28, 1] and then squeeze if needed)
        if test_image.ndim == 4:  # Assuming the shape is [1, 28, 28, 1]
            test_image = np.squeeze(test_image, axis=0)  # Now [28, 28, 1]
        if test_image.ndim == 3 and test_image.shape[-1] == 1:
            test_image = np.squeeze(test_image, axis=-1)  # Squeeze channel dimension if it's 1
        
        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)
        
        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_index)
        digit = np.argmax(output_data[0])
        prediction_digits.append(digit)

    # Compute accuracy
    accurate_count = sum(1 for i in range(len(prediction_digits)) if prediction_digits[i] == np.argmax(testY[i]))
    accuracy = accurate_count / len(prediction_digits)
    return accuracy



# Evaluate the model
accuracy = evaluate_model(interpreter, testX, testY)
print(f'Quantized model accuracy: {accuracy*100:.2f}%')
