import os
import copy
from absl import app, flags
from utils import load_model
from random_pruning import random_pruning
from dwt_pruning import wavelet_pruning

FLAGS = flags.FLAGS

"""
TODO threshold values: - might want to do sub 0.0->0.1 domain or even a deeper sub domain of 0.00->0.01
0, 0.236, 0.382, 0.5, 0.618, 0.786, 1
Decomp level increasing looks to have been a key in improving the pruning, we want to increase the level to get a
a finer image or less coarse capture of the layer, otherwise it becomes far to vulnerable to pruning. 

Practical Considerations
Level Bounds:

Minimum: 0 (no decomposition).
Maximum: Typically 3 to 5 levels for practical purposes, but it depends on the size of the input data. Larger data can be decomposed more times.
Choosing Decomposition Levels:

Higher levels of decomposition capture finer details and tend to prune fewer weights because the coefficients are smaller and subtler.
Lower levels capture coarser features, potentially resulting in more significant pruning as larger coefficients are more likely to exceed the threshold.

In PyWavelets, the upper bound for the decomposition level is determined by the size of the data 
and the specific wavelet used. Generally, 
the decomposition level cannot exceed the log base 2 of the smallest dimension of the input data.
L=⌊log2 (N/filter length - 1)⌋ 

N is the length of the data.
The filter length depends on the wavelet used.
"""
# Command line argument setup
flags.DEFINE_string('model_path', '__OGPyTorchModel__/model.safetensor',
                    'Path to the pre-trained ResNet model (bin file)')
flags.DEFINE_string('config_path', '__OGPyTorchModel__/config.json',
                    'Path to the model configuration file (.json)')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file')
flags.DEFINE_enum('wavelet', 'rbio1.3', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2'
], 'Type of discrete wavelet to use for DWT.')
flags.DEFINE_integer(
    'level', 0, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 0.1, 'Threshold value for pruning wavelet coefficients')
flags.DEFINE_string('output_dir', 'SavedModels',
                    'Directory to save the pruned models')


def main(argv):
    """
    Apply random pruning to the model based on the selective pruning log.
    This function loads a pre-trained model, applies wavelet pruning to generate a
    selectively pruned model, and then applies random pruning to create a further
    pruned model. The results are logged and saved. 
    Futere iterations might include a 3rd test were we prune based on the smallest weight.


    Args:
        selective_pruning_log (str): Path to the selective pruning log file.
        model (torch.nn.Module): The model to be pruned.
        guid (str): Unique identifier for the pruning session.
        wavelet (str): Wavelet type to be used.
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for pruning.

    Returns:
        None
    """
    model = load_model(FLAGS.model_path, FLAGS.config_path)
    print("Generating Guid")
    guid = os.urandom(4).hex()
    print("Storing Deep copy of model")
    # Create a new instance of the model for random pruning
    random_pruning_model = copy.deepcopy(model)
    print("Starting Selective purning")
    # Selective Pruning Phase
    selective_log_path = wavelet_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, FLAGS.csv_path, guid)
    print(f"Selective pruning completed. Log saved at {selective_log_path}")
    print("Starting Random purning")
    # Random Pruning Phase
    random_pruning(selective_log_path, random_pruning_model, guid,
                   FLAGS.wavelet, FLAGS.level, FLAGS.threshold, FLAGS.csv_path)


if __name__ == '__main__':
    app.run(main)
