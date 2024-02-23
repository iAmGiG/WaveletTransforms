# Simple Neural Network Model

This model serves as the foundation for our exploration into applying Discrete Wavelet Transform (DWT) for neural network model compression.

## Model Architecture

The model is a simple neural network designed for classification tasks, with the following characteristics:

- Input layer that flattens the input images.
- A hidden layer with 128 units and ReLU activation.
- An output layer with 10 units (corresponding to the number of classes) and softmax activation.

## Dataset

The model is trained and evaluated on the MNIST dataset, which consists of handwritten digits (0-9). The dataset is split into a training set for fitting the model and a test set for evaluation.

## Training

The model is trained for 10 epochs with a batch size of 32, using the Adam optimizer and categorical crossentropy as the loss function.

## Evaluation

After training, the model's performance is evaluated on the test set. This evaluation serves as the baseline for comparing the effects of applying DWT to the model's weights.

## Usage

To train and evaluate the model, simply run `simple_nn_model.py`. Ensure you have the required dependencies installed, as listed in `requirements.txt`.

## Anaconda environment setup

### Environment Creation

Create a new environment named wavelet_tf_env with Python 3.8 to ensure compatibility with TensorFlow and other libraries.

```bash
conda create --name wavelet_tf_env python=3.8
```

### Activation

Activate the newly created environment.

```bash
conda activate wavelet_tf_env
```

## Install TensorFlow (Anaconda)

Instead of using pip, we will install TensorFlow directly through Conda to manage dependencies more efficiently and potentially take advantage of automatic CUDA/cuDNN setup for GPU acceleration:

```bash
conda install -c conda-forge tensorflow
```

### Install TensorFlow

Install TensorFlow within the environment. The version of TensorFlow to install can depend on your system's capabilities (e.g., whether you have a compatible GPU and wish to use TensorFlow-GPU for faster training).

```bash
pip install tensorflow
```

### Install Additional Libraries

Install other required libraries, including those needed for handling datasets, mathematical operations, and potentially PyWavelets if you plan to integrate wavelet transforms directly in Python.

with conda environment:

```bash
conda install -c conda-forge numpy matplotlib scikit-learn pandas
```

with virtual py environment:

```bash
pip install numpy matplotlib pywavelets scikit-learn pandas huggingface_hub
```

---------------

## applying the DWT

After the model is trained and evaluated, next we'll integrate the wavelet transform process into another solution.

### The approach

- **Extract Weights:** Access the weights fro meach layer of the model that you wish to compress. within Tensorflow, this is done with 'get_weights()' method on teh layer of interest.
- **Applying DWT:** Using teh PyWavelets libray to apply DWT to the weights. We will need to decided on the wavelete type and the level of decomposition. This will yeild a set of coefficients representing the wavelet-transformed weights.
- **Thresholding:** Optionally apply thresholding to these coefficients to increase sparsity, setting small coefficients to zero. Essential for compression.
- **Inverse DWT:** applying the inverse wavelet transform (IDWT) to the modified coefficients to reconstruct teh weights in their original shape.
- **Updated model Weights:** Replace the original weights in the model with the reconstructed weights from the IDWT process.
- **Retain or Evaluate:** We may want to retain the models for evaluation as we progress.

## Model Evaluation Metrics and Methods

After training the neural network model, both with and without applying Discrete Wavelet Transform (DWT), it's crucial to evaluate and compare the model's performance based on several key metrics. Below are the metrics and methods used for this evaluation:

## 1. Accuracy

- **Purpose:** Measures the model's performance on the test dataset to assess how often it predicts the correct labels.
- **Method:** Utilize the `evaluate` function provided by TensorFlow, which returns the model's accuracy on the provided test data.

## 2. Model Size

- **Purpose:** Determines the effect of DWT on reducing the model's storage size, an important factor for deployment in resource-constrained environments.
- **Method:** Compare the file sizes of the model before and after applying DWT. This comparison is done by checking the file size of the saved model files.

## 3. Inference Time

- **Purpose:** Evaluates the time required for the model to make predictions, which is crucial for real-time applications.
- **Method:** Measure the time it takes for the model to predict labels for the test dataset, before and after DWT application. Use Python's `time` module for this measurement.

## 4. Sparsity

- **Purpose:** Quantifies the level of weight sparsity achieved through DWT, indicating the effectiveness of the model compression.
- **Method:** Calculate the ratio of zero-valued weights to the total number of weights in the model, before and after applying DWT.

## Implementation Example

Below is an example of how to implement these evaluations in Python:

```python
import time
import os
import numpy as np
from tensorflow.keras.models import load_model

# Load the model and test dataset
model = load_model('path_to_model.h5')
# Assume testX and testY are your test inputs and labels

# Accuracy
loss, accuracy = model.evaluate(testX, testY)
print(f"Accuracy: {accuracy*100:.2f}%")

# Model Size
model_size = os.path.getsize('path_to_model.h5')
print(f"Model Size: {model_size / 1024:.2f} KB")

# Inference Time
start_time = time.time()
predictions = model.predict(testX)
end_time = time.time()
print(f"Inference Time: {end_time - start_time:.4f} seconds")

# Sparsity
weights = model.get_weights()
zero_weights = np.sum([np.sum(w == 0) for w in weights])
total_weights = np.sum([w.size for w in weights])
sparsity = zero_weights / total_weights
print(f"Sparsity: {sparsity*100:.2f}%")
```
