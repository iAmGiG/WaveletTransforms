import os
import copy
import threading
import queue
from absl import app, flags
from utils import load_model, append_to_experiment_log
from random_pruning import random_pruning
from dwt_pruning import wavelet_pruning
from min_weight_pruning import min_weight_pruning
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
flags.DEFINE_string('model_path', '__OGPyTorchModel__',
                    'Path to the pre-trained ResNet model (bin file)')
flags.DEFINE_string('config_path', '__OGPyTorchModel__',
                    'Path to the model configuration file (.json)')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file')
flags.DEFINE_enum('wavelet', 'rbio1.3', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2'
                                         ], 'Type of discrete wavelet to use for DWT.')
flags.DEFINE_integer(
    'level', 1, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 1.0, 'Threshold value for pruning wavelet coefficients')
flags.DEFINE_string('output_dir', 'SavedModels',
                    'Directory to save the pruned models')

# Global queue for thread-safe logging
log_queue = queue.Queue()


def log_worker(csv_path):
    """Worker function to handle logging from multiple threads."""
    while True:
        log_entry = log_queue.get()
        if log_entry is None:
            break
        append_to_experiment_log(csv_path, *log_entry)
        log_queue.task_done()


def threaded_pruning(pruning_func, model, selective_log_path, guid, wavelet, level, threshold, csv_path, method_name, log_queue):
    """Wrapper function for threaded pruning methods."""
    try:
        result = pruning_func(selective_log_path, model, guid,
                              wavelet, level, threshold, csv_path, log_queue)
        print(f"{method_name} pruning completed.")
        return result
    except Exception as e:
        print(f"Error in {method_name} pruning: {str(e)}")
        return None


def main(argv):
    """
    Main function to run the pruning experiment with threading.

    This function loads a pre-trained model, applies wavelet pruning to generate a
    selectively pruned model, and then applies random pruning and minimum weight pruning
    in parallel threads to create further pruned models. The results are logged and saved.

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
    model_path = FLAGS.model_path
    config_path = FLAGS.config_path

    print(f"Model directory: {model_path}")
    print(f"Config file: {config_path}")

    if not os.path.isdir(model_path):
        raise ValueError(f"Provided model path {model_path} is not a valid directory.")
    
    model = load_model(model_path, config_path)
    print("Model loaded successfully.")
    print("Generating Guid")
    guid = os.urandom(4).hex()
    print(f"Generated GUID: {guid}")

    print("Storing Deep copy of model")
    # Create a new instance of the model for random pruning
    # Create deep copies of the model for different pruning methods
    dwt_model = copy.deepcopy(model)
    random_model = copy.deepcopy(model)
    min_weight_model = copy.deepcopy(model)

    # Start the logging worker thread
    log_thread = threading.Thread(target=log_worker, args=(FLAGS.csv_path,))
    log_thread.start()

    # Selective Pruning Phase (DWT)
    print("Starting Selective (DWT) Pruning")
    selective_log_path = wavelet_pruning(
        dwt_model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, FLAGS.csv_path, guid)
    print(f"Selective pruning completed. Log saved at {selective_log_path}")

    print("Starting Random purning")
    # Create and start threads for random and min weight pruning
    random_thread = threading.Thread(
        target=threaded_pruning,
        args=(random_pruning, random_model, selective_log_path, guid, FLAGS.wavelet,
            FLAGS.level, FLAGS.threshold, FLAGS.csv_path, "Random", log_queue)
    )

    print("Starting Min purning")
    min_weight_thread = threading.Thread(
        target=threaded_pruning,
        args=(min_weight_pruning, min_weight_model, selective_log_path, guid, FLAGS.wavelet,
              FLAGS.level, FLAGS.threshold, FLAGS.csv_path, "Minimum Weight", log_queue)
    )

    random_thread.start()
    min_weight_thread.start()

    # Wait for both threads to complete
    random_thread.join()
    min_weight_thread.join()

    # Signal the logging thread to finish
    log_queue.put(None)
    log_thread.join()

    print("All pruning methods completed successfully.")


if __name__ == '__main__':
    app.run(main)
