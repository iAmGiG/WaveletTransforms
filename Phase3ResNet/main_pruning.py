import uuid
from absl import app, flags
from absl.flags import FLAGS
from dwt_pruning import wavelet_pruning
from utils import load_model, save_model, setup_csv_writer

# Command line argument setup
flags.DEFINE_string('model_path', 'Phase3ResNet/__OGModel__/tf_model.h5',
                    'Path to the pre-trained ResNet model (.h5 file)')
flags.DEFINE_string('config_path', 'Phase3ResNet/__OGModel__/config.json',
                    'Path to the model configuration file (.json)')
flags.DEFINE_string('csv_path', 'pruning_log.csv', 'Path to the CSV log file')
flags.DEFINE_enum('wavelet', 'haar', ['haar', 'db1', 'db2', 'coif1', 'bior1.3', 'rbio1.3', 'sym2', 'mexh', 'morl'],
                  'Type of wavelet to use for DWT.')
flags.DEFINE_integer(
    'level', 1, 'Level of decomposition for the wavelet transform')
flags.DEFINE_float(
    'threshold', 0.5, 'Threshold value for pruning wavelet coefficients')


def main(_argv):
    model = load_model(FLAGS.model_path, FLAGS.config_path)
    csv_writer, csv_file = setup_csv_writer(FLAGS.csv_path)
    guid = str(uuid.uuid4())

    # Perform wavelet-based pruning
    pruned_model, total_prune_count = wavelet_pruning(
        model, FLAGS.wavelet, FLAGS.level, FLAGS.threshold, csv_writer, guid
    )

    save_model(pruned_model, f'pruned_model_{guid}_wavelet')
    csv_file.close()

    # Prepare for random pruning
    random_pruning_params = {
        'original_model_path': FLAGS.model_path,
        'prune_count': total_prune_count,
        'guid': guid
    }

    print(f'Run random_pruning.py with parameters: {random_pruning_params}')


if __name__ == '__main__':
    app.run(main)
