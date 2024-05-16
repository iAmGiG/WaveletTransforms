import os
import uuid
from absl import app, flags
from absl.flags import FLAGS
from dwt_pruning import wavelet_pruning
from random_pruning import random_pruning
from utils import load_model, save_model, setup_csv_writer, log_pruning_details

# Command line argument setup
flags.DEFINE_string('model_path', '__OGModel__/tf_model.h5',
                    'Path to the pre-trained ResNet model (.h5 file)')
flags.DEFINE_string('config_path', '__OGModel__/config.json',
                    'Path to the model configuration file (.json)')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file')
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2', 'mexh', 'morl'],
                  'Type of wavelet to use for DWT.')
flags.DEFINE_integer(
    'level', 1, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 0.5, 'Threshold value for pruning wavelet coefficients')


def main(_argv):
    # Load the pre-trained model
    model = load_model(FLAGS.model_path, FLAGS.config_path)

    # Create a unique identifier and setup folder structure
    guid = str(uuid.uuid4())
    pruned_instance = f"wavelet-{FLAGS.wavelet}_threshold-{FLAGS.threshold}_level-{FLAGS.level}_guid-{guid[-4:]}"
    pruned_instance_path = os.path.join('SavedModels', pruned_instance)
    selective_pruned_path = os.path.join(
        pruned_instance_path, 'selective_pruned')
    random_pruned_path = os.path.join(pruned_instance_path, 'random_pruned')

    os.makedirs(selective_pruned_path, exist_ok=True)
    os.makedirs(random_pruned_path, exist_ok=True)

    # Perform wavelet-based pruning
    selective_log_path = os.path.join(selective_pruned_path, 'log.csv')
    csv_writer, csv_file = setup_csv_writer(selective_log_path)

    pruned_model, total_prune_count = wavelet_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, csv_writer, guid
    )
    save_model(pruned_model, os.path.join(selective_pruned_path, 'model'))

    csv_file.close()

    # Perform random pruning
    random_log_path = os.path.join(random_pruned_path, 'log.csv')
    csv_writer, csv_file = setup_csv_writer(random_log_path)
    random_model = load_model(FLAGS.model_path, FLAGS.config_path)
    random_pruned_model, _ = random_pruning(
        random_model, total_prune_count, guid, csv_writer)
    save_model(random_pruned_model, os.path.join(random_pruned_path, 'model'))
    csv_file.close()


if __name__ == '__main__':
    app.run(main)
