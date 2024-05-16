import json
import tensorflow as tf
from transformers import TFResNetForImageClassification, ResNetConfig


def load_config(config_path='config.json'):
    """Load model configuration from a JSON file."""
    with open(config_path, 'r') as file:
        return json.load(file)


def main():
    """
    Dummy Input: This snippet creates a batch of random data (dummy_input) with the expected input shape 
    (modify as necessary based on your model's requirements). 
    This data is then passed to the model, effectively building it by establishing the dimensions and operations of each layer.
    Building the Model: By calling the model with the dummy_input, 
    TensorFlow constructs the internal graph for the model, which allows it to be ready for operations like loading weights 
    and summarizing.
    """
    # Load configuration
    config = load_config()

    # Create the ResNet configuration
    resnet_config = ResNetConfig(**config)

    # Create the TensorFlow version of the ResNet model
    resnet_model = TFResNetForImageClassification(resnet_config)

    # Build the model by calling it with a batch of dummy data
    input_shape = config['input_shape']
    dummy_input = tf.random.normal([1, *input_shape])  # Create dummy input with batch size 1
    resnet_model(dummy_input)  # Build the model

    # Attempt to load weights
    weights_path = 'tf_model.h5'  # Ensure this path is correct
    try:
        resnet_model.load_weights(weights_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")

    # Print the model summary to verify configuration
    print(resnet_model.summary())


if __name__ == '__main__':
    main()
