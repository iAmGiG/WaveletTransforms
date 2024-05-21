import subprocess
import os
from absl import app, flags
import utility

FLAGS = flags.FLAGS

flags.DEFINE_list('model_paths', [], 'List of paths to pruned models')


def main(argv):
    # Ensure necessary directories exist
    utility.ensure_directories_exist()

    # Step 1: Download the dataset
    subprocess.run(['python', 'download_dataset.py'])

    # Step 2: Prepare the data
    subprocess.run(['python', 'prepare_data.py'])

    # Step 3: Evaluate the models
    for model_path in FLAGS.model_paths:
        subprocess.run(['python', 'evaluate_model.py',
                       '--model_path', model_path])

    # Step 4: Visualize the results
    subprocess.run(['python', 'visualize_results.py'])

    # Step 5: Generate the report
    subprocess.run(['python', 'generate_report.py'])


if __name__ == '__main__':
    app.run(main)
