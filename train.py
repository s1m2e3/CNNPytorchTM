import os
import numpy as np
import torch
import torch.nn.functional as F
from TMCNNPytorch import TMCNNClass, TMClauses
import zipfile
import random

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

def main(config,num_clauses_construction_trials=1000, batch_per_clause_definition=100):
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
    batch_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    
    for filename in batch_files:
        batch_path = os.path.join(output_dir, filename)
        features_filters, features_colors, labels = load_batch(batch_path)
        if features_filters is None: # If load_batch failed, skip this file
            continue

        # Determine which dataset this batch belongs to
        if "train" in filename:
            dataset_key = "train"
        elif "valid" in filename:
            dataset_key = "valid"
        elif "test" in filename:
            dataset_key = "test"
        else:
            continue # Skip files that don't match
        
        data[dataset_key]["x"]['filters'].append(features_filters)
        data[dataset_key]["x"]['colors'].append(features_colors)
        data[dataset_key]["y"].append(labels)
        print(f"Loaded and appended {filename} to '{dataset_key}' dataset.")
    
    for i in range(num_clauses_construction_trials):
        # Fake binary batch: B=4, C=1, H=W=8
        # The loaded data is in NumPy format (B, H, W, C).
        batch_index = random.randrange(len(data[dataset_key]["x"]['filters']))
        numpy_batch_filters = data[dataset_key]["x"]['filters'][batch_index]
        numpy_batch_filters_keys = list(numpy_batch_filters.keys())
        numpy_batch_colors = data[dataset_key]["x"]['colors'][batch_index]
        numpy_batch_colors_keys = list(numpy_batch_colors.keys())
        numpy_batch_filters_stacked = np.concatenate(list(numpy_batch_filters.values()),axis=3)
        numpy_batch_colors_stacked = np.concatenate(list(numpy_batch_colors.values()),axis=3)
        numpy_labels = data[dataset_key]["y"][batch_index]
        
        # --- Sample a subset of the batch for this trial ---
        # First, define the random indices for sampling
        num_samples_in_batch = numpy_batch_filters_stacked.shape[0]
        sample_indices = np.random.choice(num_samples_in_batch, size=batch_per_clause_definition, replace=False)
        
        # Use the indices to collect samples from features and labels
        sampled_filters = numpy_batch_filters_stacked[sample_indices]
        sampled_colors = numpy_batch_colors_stacked[sample_indices]
        sampled_labels = numpy_labels[sample_indices]
        
        # 1) Extract literals with TMCNNClass
        kh, kw = 3, 3
        extractor = TMCNNClass()
        
        out_filters = extractor.forward(sampled_filters, kernel_size=(kh, kw))
        out_colors = extractor.forward(sampled_colors, kernel_size=(kh, kw))
        literals_filters = out_filters["literals"]
        literals_colors = out_colors["literals"]

        # # 2) Define TMClauses (e.g., 2 classes, 20 clauses/class)
        num_classes, K = 10, 200
        tm = TMClauses(num_classes, K, learn_alpha=True, init_alpha=1.0)
        tm.clause_composition(literals_filters,literals_colors, sampled_labels, literals_per_clause_proportion=0.1)
            
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