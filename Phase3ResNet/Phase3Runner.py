import os
import subprocess
import uuid
from absl import flags, app

# Define flags for potential manual overrides
FLAGS = flags.FLAGS
flags.DEFINE_string('config_file', './config.json',
                    'Path to configuration file if not using default')


def load_configuration():
    """
    Load configuration from a JSON file or use default settings from absl flags.
    """
    try:
        with open(FLAGS.config_file, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        # Load default configurations from absl flags if no config file is present
        return {
            'model_directory': os.getcwd(),
            'log_directory': os.getcwd(),
            'wavelet_type': FLAGS.wavelet_type,
            'level': FLAGS.level,
            'threshold': FLAGS.threshold,
        }


def find_latest_file(directory, extension):
    """
    Find the latest file with the specified extension in the given directory.
    """
    list_of_files = [os.path.join(directory, f) for f in os.listdir(
        directory) if f.endswith(extension)]
    if list_of_files:
        return max(list_of_files, key=os.path.getctime)
    return None


def generate_guid():
    """
    Generate a unique GUID for each test run.
    """
    return str(uuid.uuid4())


def check_if_model_was_tested(wavelet, level, threshold, model_directory):
    """
    Check if a model with specific parameters has already been tested.
    """
    pattern = f"*_{wavelet}_{level}_{threshold}.h5"
    for filename in os.listdir(model_directory):
        if fnmatch.fnmatch(filename, pattern):
            return os.path.join(model_directory, filename)
    return None


def execute_script(script_path, args):
    """
    Execute a Python script with the specified arguments.
    """
    subprocess.run(['python', script_path] + args, check=True)


def main():
    """
    Main execution flow of the runner script.
    """
    config = load_configuration()
    model_path = find_latest_file(config['model_directory'], '.h5')
    if model_path is None:
        raise Exception("No model file found.")

    guid = generate_guid()
    wavelet_type = config.get('wavelet_type', 'haar')
    level = config.get('level', 1)
    threshold = config.get('threshold', 0.5)

    pruned_model_path = check_if_model_was_tested(
        wavelet_type, level, threshold, config['model_directory'])
    if pruned_model_path is None:
        # Execute pruning script
        execute_script('DWTResNetPruning.py', [
                       model_path, wavelet_type, str(level), str(threshold), guid])

    # Assuming the pruned model file path is known
    execute_script('TestingScript.py', [pruned_model_path])


if __name__ == "__main__":
    app.run(main)
