import time
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load or prepare your test dataset
(testX, testY), (_, _) = mnist.load_data()
testX = testX / 255.0  # Normalize
testY = to_categorical(testY)

# make this line abstract based on flags.
model_path = './DeepLearning/SavedStandardModels/mnist_model_02-20_17-44.h5'

# Load your model
model = load_model(model_path)


# Evaluate accuracy
loss, accuracy = model.evaluate(testX, testY)
print(f"Accuracy: {accuracy*100:.2f}%")

# Evaluate model size
model_size = os.path.getsize(model_path)
print(f"Model Size: {model_size / 1024:.2f} KB")

# Evaluate inference time
start_time = time.time()
predictions = model.predict(testX)
end_time = time.time()
print(f"Inference Time: {end_time - start_time:.4f} seconds")

# Evaluate sparsity
weights = model.get_weights()
zero_weights = np.sum([np.sum(w == 0) for w in weights])
total_weights = np.sum([w.size for w in weights])
sparsity = zero_weights / total_weights
print(f"Sparsity: {sparsity*100:.2f}%")


"""
model 1 and trained with mnist.
model 2 but quantized with dwt.
now what about the paramater size?
what is the parma size?
with the mnist,
now have a threashold for the quantization, give this,
lets quantize more on the param sizes,
repeat thsi and increase the streangh of the reduciton
then find the point where the therashold pushes 

where does the reduction threashold push the accuracy off the cliff.
evaluate the size and contnue to explore the 

param size vs the performance, 
find any critical points of % reduction.

what is the performance chagne after x% change?
accuracy, ...ect.

now we can measure the performnce of proof of concept, 
the take on the BERT, or ResNet-pretrained
then do this on the text and image.

how much size reduction can we do before we end up reducing to much to not enough?
DWT matrix reduction, and reducing more, reconstruct back, and reconstruct the size.
then performance test.

relation between performnace and size reduction.
then far away is the adaptive learning process.'
put this in a chart, make the 
PURPOSE: and then we can improve the vision then reduce the size for mobile/edge devices.

"""