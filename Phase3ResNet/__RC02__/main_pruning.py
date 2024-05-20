import numpy as np
import tensorflow as tf
import pywt
import os
import csv
import uuid
from absl import app, flags
from transformers import TFResNetForImageClassification

FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', 'microsoft/resnet-18',
                    'Name of the pretrained model')
flags.DEFINE_string('csv_path', 'experiment_log.csv',
                    'Path to the CSV log file')
flags.DEFINE_string('wavelet', 'haar', 'Wavelet name')
flags.DEFINE_integer('level', 0, 'Wavelet decomposition level')
flags.DEFINE_float('threshold', 0.1, 'Threshold value for pruning')


def load_model(model_name):
    model = TFResNetForImageClassification.from_pretrained(model_name)
    return model


def dwt_pruning(model, wavelet_name, level, threshold):
    pruned_count = {}
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            weights = [layer.kernel.numpy()]
        else:
            weights = layer.get_weights()

        if not weights:
            continue

        pruned_weights = []
        total_pruned = 0
        for weight in weights:
            coeffs = pywt.wavedec(weight.flatten(), wavelet_name, level=level)
            coeffs_thresh = [pywt.threshold(
                c, threshold, mode='soft') for c in coeffs]
            pruned_weight = pywt.waverec(coeffs_thresh, wavelet_name)
            pruned_weight = pruned_weight[:weight.size].reshape(weight.shape)
            total_pruned += np.sum(pruned_weight == 0)
            pruned_weights.append(pruned_weight)

        if hasattr(layer, 'kernel'):
            layer.kernel.assign(pruned_weights[0])
        else:
            layer.set_weights(pruned_weights)

        pruned_count[layer.name] = total_pruned
        print(f"Layer {layer.name} pruned. Total pruned count: {total_pruned}")

    return model, pruned_count


def random_pruning(model, layer_prune_counts):
    total_pruned = 0

    def prune_layer(layer, layer_name):
        nonlocal total_pruned
        if hasattr(layer, 'kernel'):
            weights = [layer.kernel.numpy()]
        else:
            weights = layer.get_weights()

        if not weights:
            return

        pruned_weights = []
        for weight in weights:
            flat_weights = weight.flatten()
            prune_count = layer_prune_counts.get(layer_name, 0)

            if prune_count == 0:
                continue

            prune_indices = np.random.choice(
                flat_weights.size, prune_count, replace=False)
            flat_weights[prune_indices] = 0
            pruned_weights.append(flat_weights.reshape(weight.shape))

        if pruned_weights:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(pruned_weights[0])
            else:
                layer.set_weights(pruned_weights)

            total_pruned += prune_count

    def recursive_prune(current_layer, layer_path=""):
        layer_name = f"{layer_path}/{current_layer.name}" if layer_path else current_layer.name
        if isinstance(current_layer, tf.keras.Model):
            for sub_layer in current_layer.layers:
                recursive_prune(sub_layer, layer_path=layer_name)
        elif isinstance(current_layer, tf.keras.layers.Layer):
            prune_layer(current_layer, layer_name)

    recursive_prune(model)
    print(f"Total weights pruned: {total_pruned}")
    return model


def save_model(model, path):
    model.save(path)
    print(f"Model saved successfully at {path}")


def log_pruning_details(csv_path, guid, wavelet, level, threshold, pruning_phase, layer_prune_counts):
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for layer_name, pruned_count in layer_prune_counts.items():
            writer.writerow([guid, wavelet, level, threshold,
                            pruning_phase, layer_name, pruned_count])
    print(f"Logged pruning details in {csv_path}")


def main(argv):
    del argv  # Unused
    model_name = FLAGS.model_name
    csv_path = FLAGS.csv_path
    wavelet = FLAGS.wavelet
    level = FLAGS.level
    threshold = FLAGS.threshold

    guid = str(uuid.uuid4())

    # Load the model
    model = load_model(model_name)

    # Perform DWT pruning
    model, dwt_prune_counts = dwt_pruning(model, wavelet, level, threshold)
    dwt_save_path = f"SavedModels/{wavelet}_threshold-{threshold}_level-{level}_guid-{guid}/selective_pruned"
    os.makedirs(dwt_save_path, exist_ok=True)
    save_model(model, dwt_save_path)
    log_pruning_details(csv_path, guid, wavelet, level,
                        threshold, 'selective', dwt_prune_counts)

    # Perform Random pruning
    random_pruned_model = random_pruning(model, dwt_prune_counts)
    random_save_path = f"SavedModels/{wavelet}_threshold-{threshold}_level-{level}_guid-{guid}/random_pruned"
    os.makedirs(random_save_path, exist_ok=True)
    save_model(random_pruned_model, random_save_path)
    log_pruning_details(csv_path, guid, wavelet, level,
                        threshold, 'random', dwt_prune_counts)


if __name__ == '__main__':
    app.run(main)
