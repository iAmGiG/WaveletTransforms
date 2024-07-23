import os
import logging
import traceback
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from absl import app, flags
from eval_model import evaluate_model
from setup_test_dataloader import prepare_validation_dataloader
from utils import load_model

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', '../SavedModels/haar_threshold-0.236_level-3_guid-f916',
                    'Path to the parent model directory')
flags.DEFINE_string('data_path', 'imagenet1k/data/val_images',
                    'Path to the ImageNet validation data')
flags.DEFINE_integer('batch_size', 64, 'Batch size for the DataLoader.')
flags.DEFINE_boolean('gpu', True, 'Whether to use GPU for model evaluation.')
flags.DEFINE_integer(
    'num_threads', 3, 'Number of concurrent threads to use for evaluation.')
flags.DEFINE_integer(
    'timeout', 3600, 'Timeout in seconds for each model evaluation')


def evaluate_model_wrapper(model_dir, val_loader, device):
    try:
        model_name = os.path.basename(model_dir)
        logging.info(f"Starting evaluation for model: {model_name}")
        
        model = load_model(model_dir)
        if model is None:
            logging.error(f"Failed to load model from {model_dir}")
            return model_name, None, None, None, None
        
        model.to(device)
        
        accuracy, f1, recall, cm, avg_loss = evaluate_model(model, val_loader, device)
        
        logging.info(f"Evaluation complete for {model_name}. Metrics:")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Recall: {recall}")
        logging.info(f"Average Loss: {avg_loss}")
        
        # Save metrics in the model's own directory
        metrics_path = os.path.join(model_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Average Loss: {avg_loss}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
        
        logging.info(f"Results saved for {model_name} in {metrics_path}")
        
        return model_name, accuracy, f1, recall, avg_loss
    
    except Exception as e:
        logging.error(f"An error occurred during evaluation of {model_name}: {str(e)}")
        logging.error(traceback.format_exc())
        return model_name, None, None, None, None


def main(argv):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Device configuration
        device = torch.device(
            "cuda" if torch.cuda.is_available() and FLAGS.gpu else "cpu")

        # Prepare validation dataloader
        val_loader, val_dataset = prepare_validation_dataloader(
            FLAGS.data_path, batch_size=FLAGS.batch_size)
        logging.info(f"Validation dataset size: {len(val_dataset)}")

        # Get subdirectories
        model_dirs = [os.path.join(FLAGS.model_path, d) for d in os.listdir(FLAGS.model_path)
                      if os.path.isdir(os.path.join(FLAGS.model_path, d))]

        results = []
        # Use ThreadPoolExecutor for concurrent evaluation
        with ThreadPoolExecutor(max_workers=FLAGS.num_threads) as executor:
            future_to_model = {executor.submit(
                evaluate_model_wrapper, model_dir, val_loader, device): model_dir for model_dir in model_dirs}

            for future in as_completed(future_to_model):
                model_dir = future_to_model[future]
                try:
                    result = future.result(timeout=FLAGS.timeout)
                    if result[1] is not None:  # Check if evaluation was successful
                        results.append(result)
                    else:
                        logging.error(
                            f"Evaluation failed for model in directory: {model_dir}")
                except TimeoutError:
                    logging.error(
                        f"Evaluation timed out for model in directory: {model_dir}")
                except Exception as exc:
                    logging.error(
                        f"Evaluation generated an exception for model in directory {model_dir}: {exc}")

        # Summarize results
        logging.info("Evaluation complete for all models. Summary:")
        for model_name, accuracy, f1, recall, avg_loss in results:
            logging.info(f"Model: {model_name}")
            logging.info(f"  Accuracy: {accuracy}")
            logging.info(f"  F1 Score: {f1}")
            logging.info(f"  Recall: {recall}")
            logging.info(f"  Average Loss: {avg_loss}")
            logging.info("--------------------")

    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    app.run(main)
