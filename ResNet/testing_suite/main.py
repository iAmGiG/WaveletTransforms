from absl import app, flags
import logging
import traceback
import torch
from eval_model import evaluate_model
from setup_test_dataloader import prepare_validation_dataloader
from utils import load_model
import os

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', '../__OGPyTorchModel__', 'Path to the model directory')
flags.DEFINE_string('data_path', 'imagenet1k/data/val_images', 'Path to the ImageNet validation data')
flags.DEFINE_integer('batch_size', 64, 'Batch size for the DataLoader.')
flags.DEFINE_string('output_dir', '../__OGPyTorchModel__', 'Directory where the output results should be saved.')
flags.DEFINE_boolean('gpu', True, 'Whether to use GPU for model evaluation.')


def main(argv):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() and FLAGS.gpu else "cpu")

        # Load pre-trained model from the specified model path using utils.load_model
        model = load_model(FLAGS.model_path)
        model.to(device)
        logging.info(f"Model loaded and moved to device: {device}")

        # Prepare validation dataloader using the specified data path
        val_loader, val_dataset = prepare_validation_dataloader(FLAGS.data_path, batch_size=FLAGS.batch_size)
        logging.info(f"Validation dataset size: {len(val_dataset)}")

        # Evaluate model
        logging.info("Starting model evaluation")
        accuracy, f1, recall, cm = evaluate_model(model, val_loader, device)

        logging.info(f"Evaluation complete. Metrics:")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Recall: {recall}")

        # Ensure output directory exists
        os.makedirs(FLAGS.output_dir, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(FLAGS.output_dir, 'initial_evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

        logging.info("Evaluation complete. Results saved.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    app.run(main)
