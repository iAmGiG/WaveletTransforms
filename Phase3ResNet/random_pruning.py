import numpy as np
import tensorflow as tf
import uuid
from utils import load_model, save_model, setup_csv_writer, log_pruning_details


def random_pruning(model, prune_count):
    total_pruned = 0
    for layer in model.layers:
        if layer.trainable and layer.get_weights():
            weights = layer.get_weights()
            pruned_weights = []
            for weight in weights:
                flat_weights = weight.flatten()
                prune_indices = np.random.choice(
                    flat_weights.size, prune_count, replace=False)
                flat_weights[prune_indices] = 0
                pruned_weights.append(flat_weights.reshape(weight.shape))
            layer.set_weights(pruned_weights)
            total_pruned += prune_count
    return model, total_pruned


def main(original_model_path, prune_count, guid):
    model = load_model(original_model_path,
                       original_model_path.replace('.h5', '_config.json'))
    csv_writer, csv_file = setup_csv_writer('random_pruning_log.csv')

    pruned_model, total_pruned = random_pruning(model, prune_count)
    log_pruning_details(csv_writer, guid, 'N/A', 'N/A', 'N/A',
                        'random', 'N/A', 'N/A', total_pruned, 'N/A')

    save_model(pruned_model, f'pruned_model_{guid}_random')
    csv_file.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python random_pruning.py <original_model_path> <prune_count> <guid>")
        sys.exit(1)
    original_model_path = sys.argv[1]
    prune_count = int(sys.argv[2])
    guid = sys.argv[3]
    main(original_model_path, prune_count, guid)
