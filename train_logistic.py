import os
import numpy as np
import torch
import torch.nn.functional as F
from TMCNNPytorch import TMCNNClass, TMClauses
from sklearn.linear_model import LogisticRegression
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import yaml

def load_batch(batch_path):
    """
    Loads a batch of features and labels from a .npz file.
    Concatenates all feature arrays into a single array.
    """
    try:
        loaded_data = np.load(batch_path, allow_pickle=False)
    except (zipfile.BadZipFile, ValueError) as e:
        print(f"Warning: Could not load batch file '{batch_path}'. It may be corrupted. Skipping. Error: {e}")
        return None, None

    # Get all keys except for labels and metadata
    feature_keys = [k for k in loaded_data.keys() if k not in ['labels', '_metadata']]
    
    # Load and concatenate all feature arrays along the channel axis (axis=3)
    # This creates a single large feature map per image.
    all_features_filters = {key:loaded_data[key]  for key in feature_keys if 'filters' in key}
    all_features_colors = {key:loaded_data[key]  for key in feature_keys if 'kmeans' in key}
    
    labels = loaded_data['labels']
    
    return all_features_filters, all_features_colors, labels

def run_logistic_regression_on_subset(features, labels, num_features_to_sample=1000):
    """
    Runs logistic regression on a subset of features to find important ones.
    This is a helper function for process_trial.
    """
    num_samples, num_total_features = features.shape
    num_classes = labels.shape[1]
    
    y_indices = np.argmax(labels, axis=1)

    log_reg = LogisticRegression(multi_class='ovr', solver='lbfgs', penalty="l2", C=0.1, max_iter=100, random_state=None)
    
    feature_indices = np.random.choice(num_total_features, size=num_features_to_sample, replace=False)
    x_subset = features[:, feature_indices]
    
    try:
        log_reg.fit(x_subset, y_indices)
        accuracy = log_reg.score(x_subset, y_indices)
        abs_coeffs = np.abs(log_reg.coef_).mean(axis=0)
        quantile_95 = np.quantile(abs_coeffs, 0.95)
        print(f"  - LogReg Accuracy: {accuracy:.4f}, 95th Quantile Threshold: {quantile_95:.4f}")
        relevant_features_in_subset = np.where(abs_coeffs >= quantile_95)[0]
        
        # Map back to original feature indices
        original_indices = feature_indices[relevant_features_in_subset]
        return original_indices.tolist()
    except Exception:
        return []

def process_trial(data, dataset_key, batch_per_clause_definition, kh, kw, extractor,relevant_indices):
    """
    A function that encapsulates the logic for a single trial, suitable for parallel execution.
    It now returns the indices of relevant literals found in the trial.
    """
    # The loaded data is in NumPy format (B, H, W, C).
    batch_index = random.randrange(len(data[dataset_key]["x"]['filters']))
    numpy_batch_filters = data[dataset_key]["x"]['filters'][batch_index]
    numpy_batch_colors = data[dataset_key]["x"]['colors'][batch_index]
    
    numpy_batch_filters_stacked = np.concatenate(list(numpy_batch_filters.values()), axis=3)
    numpy_batch_colors_stacked = np.concatenate(list(numpy_batch_colors.values()), axis=3)
    numpy_labels = data[dataset_key]["y"][batch_index]
    
    # --- Sample a subset of the batch for this trial ---
    num_samples_in_batch = numpy_batch_filters_stacked.shape[0]
    if batch_per_clause_definition > num_samples_in_batch:
        print(f"Warning: batch_per_clause_definition ({batch_per_clause_definition}) is larger than the number of samples in the batch ({num_samples_in_batch}). Using all samples.")
        sample_indices = np.arange(num_samples_in_batch)
    else:
        sample_indices = np.random.choice(num_samples_in_batch, size=batch_per_clause_definition, replace=False)

    # Use the indices to collect samples from features and labels
    sampled_filters = numpy_batch_filters_stacked[sample_indices]
    sampled_colors = numpy_batch_colors_stacked[sample_indices]
    sampled_labels = numpy_labels[sample_indices]
    
    out_filters = extractor.forward(sampled_filters, kernel_size=(kh, kw))
    out_colors = extractor.forward(sampled_colors, kernel_size=(kh, kw))
    literals_filters = out_filters["literals"]
    literals_colors = out_colors["literals"]
    
    # Combine and flatten literals for logistic regression
    x_combined_np = np.concatenate([literals_filters, literals_colors], axis=1)
    flattened_x = x_combined_np.reshape(x_combined_np.shape[0], -1)
    
    # One-hot encode labels
    num_classes = 10 # Assuming CIFAR-10
    y_squeezed = np.squeeze(sampled_labels).astype(int)
    one_hot_labels = np.zeros((y_squeezed.shape[0], num_classes), dtype=np.uint8)
    one_hot_labels[np.arange(y_squeezed.shape[0]), y_squeezed] = 1
    
    # Run logistic regression multiple times on different feature subsets
    all_found_indices = []
    for _ in range(20): # 20 trials per batch
        all_found_indices.extend(run_logistic_regression_on_subset(flattened_x, one_hot_labels))
        
    return all_found_indices
def main(config,num_clauses_construction_trials=1000, batch_per_clause_definition=50):
    # --- 1. Load all preprocessed data from the 'outputs' directory ---
    print("\n--- Loading preprocessed data ---")
    output_dir = "outputs"
    
    # Initialize dictionaries to hold the data
    data = {
        "train": {"x": {"filters": [], "colors": []}, "y": []},
        "valid": {"x": {"filters": [], "colors": []}, "y": []},
        "test": {"x": {"filters": [], "colors": []}, "y": []}
    }

    # Find all .npz files in the output directory
    batch_files = [f for f in os.listdir(output_dir) if f.endswith('.npz') and 'train' in f]
    
    for filename in batch_files:
        batch_path = os.path.join(output_dir, filename)
        features_filters, features_colors, labels = load_batch(batch_path)
        if features_filters is None: # If load_batch failed, skip this file
            continue
        
        # Since we are only loading 'train' files, we can directly use the 'train' key
        dataset_key = "train"
        data[dataset_key]["x"]['filters'].append(features_filters)
        data[dataset_key]["x"]['colors'].append(features_colors)
        data[dataset_key]["y"].append(labels)
        print(f"Loaded and appended {filename} to '{dataset_key}' dataset.")
    
    # 1) Extract literals with TMCNNClass
    kh, kw = 3, 3
    extractor = TMCNNClass()

    # # 2) Define TMClauses (e.g., 2 classes, 20 clauses/class)
    num_classes, K = 10, 200
    tm = TMClauses(num_classes, K, learn_alpha=True, init_alpha=1.0)

    # Run trials sequentially for easier debugging
    print(f"\n--- Starting {num_clauses_construction_trials} clause construction trials sequentially ---")
    
    all_relevant_indices = []
    for i in range(num_clauses_construction_trials):
        try:
            print(f"Starting trial {i+1}/{num_clauses_construction_trials}...", flush=True)
            trial_indices = process_trial(data, dataset_key, batch_per_clause_definition, kh, kw, extractor,all_relevant_indices)
            filtered =[index for index in trial_indices if index not in all_relevant_indices]
            all_relevant_indices.extend(filtered)
            print(f"Completed trial {i+1}/{num_clauses_construction_trials}. Found {len(filtered)} new literals. Currently there are: {len(all_relevant_indices)} relevant literals.", flush=True)
        except Exception as e:
            # With sequential execution, the traceback will be much clearer.
            print(f"Trial {i+1}/{num_clauses_construction_trials} failed with an error: {e}")
        if len(all_relevant_indices) >= 2e5:
            break
    # Aggregate results in the main process
    unique_indices = sorted(list(set(all_relevant_indices)))
    tm.relevant_indices = unique_indices
    print(f"\nClause construction finished. Found a total of {len(unique_indices)} unique relevant literals.")

    # --- Save unique indices to YAML file ---
    yaml_path = os.path.join(output_dir, "index.yaml")
    print(f"Saving {len(unique_indices)} unique indices to {yaml_path}...")
    with open(yaml_path, 'w') as f:
        yaml.dump(unique_indices, f)
    print("Save complete.")
    # # Concatenate all batches for each dataset
    # for key in data:
    #     if data[key]["x"]: # Ensure there's data to concatenate
        #         data[key]["x"] = np.concatenate(data[key]["x"], axis=0)
        #         data[key]["y"] = np.concatenate(data[key]["y"], axis=0)
        #         print(f"Final shape of {key} data (X): {data[key]['x'].shape}")
        #         print(f"Final shape of {key} labels (y): {data[key]['y'].shape}")

    

if __name__ == "__main__":
    # The config would be loaded from a YAML file in a real scenario
    # For this example, we pass an empty dictionary.
    main(config={})