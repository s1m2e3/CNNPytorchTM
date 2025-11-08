import os
import numpy as np
import torch
import torch.nn.functional as F
from TMCNNPytorch import TMCNNClass, TMClauses
import zipfile

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
    print(feature_keys)
    all_features = [loaded_data[key] for key in feature_keys]
    
    labels = loaded_data['labels']
    
    return all_features, labels

def main(config):
    # --- 1. Load all preprocessed data from the 'outputs' directory ---
    print("\n--- Loading preprocessed data ---")
    output_dir = "outputs"
    
    # Initialize dictionaries to hold the data
    data = {
        "train": {"x": [], "y": []},
        "valid": {"x": [], "y": []},
        "test": {"x": [], "y": []}
    }

    # Find all .npz files in the output directory
    batch_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    print(batch_files)
    input('hipi')
    for filename in batch_files:
        batch_path = os.path.join(output_dir, filename)
        features, labels = load_batch(batch_path)
        if features is None: # If load_batch failed, skip this file
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

        data[dataset_key]["x"].append(features)
        data[dataset_key]["y"].append(labels)
        print(f"Loaded and appended {filename} to '{dataset_key}' dataset.")
        
        # Fake binary batch: B=4, C=1, H=W=8
        # The loaded data is in NumPy format (B, H, W, C).
        numpy_batch = data[dataset_key]["x"][0][0]
        B, H, W, C = numpy_batch.shape
        
        # 1) Extract literals with TMCNNClass
        kh, kw = 3, 3
        extractor = TMCNNClass(in_channels=C, kernel_size=(kh, kw))

        # Convert to a PyTorch tensor and permute to (B, C, H, W) for the model.
        torch_batch = torch.from_numpy(numpy_batch).permute(0, 3, 1, 2)
        torch_labels = torch.from_numpy(data[dataset_key]["y"][0])
        out = extractor(torch_batch)
        literals = out["literals"]        # [B, 2*C*kh*kw, L]
        D = literals.shape[1] # D is the number of literals, which is the size of the second dimension.
        print(f"Input shape (B, H, W, C): {numpy_batch.shape}")
        print(f"Literals shape (B, D, L): {literals.shape}")
        input('hipi')
        # # 2) Define TMClauses (e.g., 2 classes, 20 clauses/class)
        num_classes, K = 10, 200
        tm = TMClauses(num_classes, K, learn_alpha=True, init_alpha=1.0)
        tm.clause_composition(literals, torch_labels, literals_per_clause_proportion=0.1)
        
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