from pre_processing_tools import create_and_fit_processors
import tensorflow as tf

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    import yaml
    import os

    # --- Load configuration from YAML ---
    # Assumes the script is run from the project root (e.g., CNNPytorchTM/)
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    cifar = tf.keras.datasets.cifar10    
    (x_train_full, y_train_full), (x_test, y_test) = cifar.load_data()
    # Do not normalize here; ToTensor() will handle it.
    x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    # --- Create and fit the processors ONCE on the training data ---
    kmeans_model, filter_bank, binarizer = create_and_fit_processors(x_train, config)

    # --- Extract learned thresholds and save them back to the config file ---
    print("\n--- Updating config.yaml with learned thresholds ---")
    if binarizer.thr_filters_ is not None:
        # Convert numpy array to a standard list for YAML serialization
        config['binarizer']['thresholds_filters'] = binarizer.thr_filters_.tolist()
    if binarizer.thr_kmeans_ is not None:
        config['binarizer']['thresholds_kmeans'] = binarizer.thr_kmeans_.tolist()

    # --- Extract learned K-Means centroids and save them as well ---
    if hasattr(kmeans_model, 'centroids_') and kmeans_model.centroids_ is not None:
        print("Adding learned K-Means centroids to config.")
        config['kmeans']['centroids'] = kmeans_model.centroids_.tolist()

    # Write the updated config back to the file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Successfully saved thresholds to {config_path}")
