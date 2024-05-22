import subprocess
from absl import app, flags
import utility
import os

FLAGS = flags.FLAGS

flags.DEFINE_list('model_paths', [], 'List of paths to pruned models')


def main(argv):
    # Ensure necessary directories exist
    utility.ensure_directories_exist()

    # Step 1: Check if the dataset subset is cached
    if not os.path.exists('imagenet-1k-small-test'):
        subprocess.run(['python', 'download_subset.py'])
    else:
        print("Dataset subset already cached.")

    # Step 2: Prepare the data
    subprocess.run(['python', 'prepare_data.py'])

    # Step 3: Evaluate the models
    model_paths = FLAGS.model_paths or [
        '../__OGModel__/tf_model.h5',
        '../__OGPyTorchModel__/pytorch_model.bin'
    ]

    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Evaluating model: {model_path}")
            subprocess.run(['python', 'evaluate_model.py',
                           '--model_path', model_path])
        else:
            print(f"Model path {model_path} does not exist. Skipping.")

    # Step 4: Visualize the results
    subprocess.run(['python', 'visualize_results.py'])

    # Step 5: Generate the report
    subprocess.run(['python', 'generate_report.py'])


if __name__ == '__main__':
    app.run(main)
