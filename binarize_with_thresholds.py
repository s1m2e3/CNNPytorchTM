from pre_processing_tools import KMeansColorQuantizer, ClassicFilterBankNP, Binarizer, transform_batch, save_features_with_metadata_npz, GaussianSpec, LaplacianSpec, GaborSpec
import tensorflow as tf
import numpy as np

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

    # --- Instantiate processors directly from config (no fitting) ---
    print("\n--- Initializing processors from config file ---")
    
    # 1. K-Means: Pass the kmeans part of the config. It will load centroids if they exist.
    kmeans_cfg = config['kmeans']
    kmeans_model = KMeansColorQuantizer(config=kmeans_cfg, K=kmeans_cfg['num_centroids'])

    # 2. Filter Bank: Initialize with its configuration.
    fb_cfg = config['filter_bank']
    gs = [GaussianSpec(sigma=s) for s in fb_cfg['gaussian']['sigmas']]
    ls = [LaplacianSpec(sigma=s) for s in fb_cfg['laplacian']['sigmas']]
    thetas = np.linspace(0, np.pi, fb_cfg['gabor']['num_thetas'], endpoint=False)
    bs = [GaborSpec(theta=t, lambd=fb_cfg['gabor']['lambd']) for t in thetas]
    filter_bank = ClassicFilterBankNP(gs, ls, bs, per_channel=fb_cfg['per_channel'], padding=fb_cfg['padding'], batch_size=fb_cfg.get('processing_batch_size', 32))

    # 3. Binarizer: Pass the binarizer part of the config. It will load thresholds if they exist.
    binarizer = Binarizer(config=config.get('binarizer'))

    datasets_to_process = {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test}
    labels_to_process = {'x_train': y_train, 'x_valid': y_valid, 'x_test': y_test}
    batch_size = 1000
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, data_array in datasets_to_process.items():
        label_array = labels_to_process[dataset_name]
        num_samples = data_array.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        print(f"\n--- Processing dataset: {dataset_name} in {num_batches} batches ---")
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_data = data_array[start_idx:end_idx]
            batch_labels = label_array[start_idx:end_idx]
            
            print(f"Processing batch {i+1}/{num_batches} for {dataset_name} (samples {start_idx}-{end_idx-1})...")
            final_features = transform_batch(batch_data, kmeans_model, filter_bank, binarizer)
            
            # Add the corresponding labels to the dictionary before saving
            final_features['labels'] = batch_labels
            
            output_filename = f"features_with_metadata_{dataset_name}_batch_{i+1}.npz"
            output_path = os.path.join(output_dir, output_filename)
            
            save_features_with_metadata_npz(final_features, output_path, metadata=config)
            print(f"Batch {i+1} processing complete. Saved to {output_path}")


    # # --- Example of how to load the data, metadata, and labels ---
    # print("\n--- Loading and verifying the last saved file ---")
    # loaded_data = np.load(output_path, allow_pickle=True)
    
    # # Retrieve the metadata
    # # The metadata is a 0-d array; use .item() to get the dictionary back
    # loaded_metadata = loaded_data['_metadata'].item()
    
    # print("Successfully loaded metadata. K-Means config from metadata:")
    # print(loaded_metadata['kmeans'])

    # # You can now access your feature arrays AND the labels
    # gaussian_low_band = loaded_data['filters_gaussian_low']
    # saved_labels = loaded_data['labels']
    # print(f"\nShape of loaded 'filters_gaussian_low': {gaussian_low_band.shape}")
    # print(f"Shape of loaded 'labels': {saved_labels.shape}")
    # print(f"First 5 labels: {saved_labels[:5].flatten()}")