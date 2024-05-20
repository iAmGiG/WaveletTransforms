import unittest
import numpy as np
import tensorflow as tf
from transformers import TFResNetForImageClassification
from dummy_pruning import load_saved_model, dummy_random_pruning, dummy_prune_counts


class TestRandomPruning(unittest.TestCase):

    def setUp(self):
        # Load the pre-saved ResNet-18 model
        model_path = '__OGModel__/tf_model.h5'
        config_path = '__OGModel__/config.json'
        self.model = load_saved_model(model_path, config_path)

        # Define dummy prune counts
        self.dummy_prune_counts = dummy_prune_counts

    def test_pruning(self):
        # Get the original weights before pruning
        original_weights = {layer.name: layer.get_weights()
                            for layer in self.model.layers}

        # Debug: Print original weights
        print("Original weights:")
        for layer_name, weights in original_weights.items():
            print(
                f"Layer: {layer_name}, Weights shape: {[w.shape for w in weights]}")

        # Apply the dummy random pruning
        pruned_model, total_pruned = dummy_random_pruning(
            self.model, self.dummy_prune_counts)

        # Debug: Print pruned weights
        pruned_weights = {layer.name: layer.get_weights()
                          for layer in pruned_model.layers}
        print("Pruned weights:")
        for layer_name, weights in pruned_weights.items():
            print(
                f"Layer: {layer_name}, Weights shape: {[w.shape for w in weights]}")

        # Check that the total pruned weights match the expected number
        expected_total_pruned = sum(self.dummy_prune_counts.values())
        self.assertEqual(total_pruned, expected_total_pruned,
                         "Total pruned weights do not match expected number")

        # Check that the correct number of weights are pruned for each layer
        for layer in pruned_model.layers:
            layer_name = f"{layer.name}"
            if layer_name in self.dummy_prune_counts:
                original_weights_flat = np.concatenate(
                    [w.flatten() for w in original_weights[layer.name]])
                pruned_weights_flat = np.concatenate(
                    [w.flatten() for w in layer.get_weights()])

                pruned_count = np.sum(
                    original_weights_flat != pruned_weights_flat)
                self.assertEqual(pruned_count, self.dummy_prune_counts[layer_name],
                                 f"Pruned weights for layer {layer_name} do not match expected number")


if __name__ == "__main__":
    unittest.main()
