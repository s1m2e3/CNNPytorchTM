import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# TMCNNClass
# -----------------------------------------------------------------------------
class TMCNNClass:
    r"""
    Convolutional patch/literal extractor for Tsetlin-style pipelines using NumPy.

    IMPORTANT: Inputs **must be binary** tensors in {0,1}.

    What it does
    ------------
    - Extracts all sliding patches of size (kh, kw) over C channels.
    - Produces *all possible literals* per patch:
        * positive literals  : x
        * negative literals  : 1 - x
      packed along the "literal" dimension.

    Input
    -----
    - x: NumPy array of shape [B, H, W, C] with values in {0,1}.

    Output
    ------
    Returns a dict with:
      - 'literals': [B, 2*C*kh*kw, L] uint8 in {0,1} (L = number of patches)
      - 'spatial': (H_out, W_out)
      - 'meta': {'C': C, 'kh': kh, 'kw': kw}
    """
    def __init__(self):
        pass

    def forward(self, x: np.ndarray, kernel_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        x: NumPy array of shape [B, H, W, C] with values in {0,1}.
        kernel_size: Tuple (kh, kw) for the patch dimensions.
        """
        assert isinstance(x, np.ndarray), "Input must be a NumPy array."
        assert x.ndim == 4, f"Input must be a 4D array [B, H, W, C], but got {x.ndim} dimensions."
        B, H, W, C = x.shape
        kh, kw = kernel_size

        x_uint8 = x.astype(np.uint8)

        H_out = H - kh + 1
        W_out = W - kw + 1
        L = H_out * W_out

        # Use stride tricks for efficient patch extraction (no data duplication)
        shape = (B, H_out, W_out, C, kh, kw)
        strides = x_uint8.strides
        new_strides = (strides[0], strides[1], strides[2], strides[3], strides[1], strides[2])
        
        # Create a view of patches: [B, H_out, W_out, C, kh, kw]
        patches = np.lib.stride_tricks.as_strided(x_uint8, shape=shape, strides=new_strides)

        # Reshape to get positive literals: [B, C*kh*kw, L]
        pos = patches.transpose(0, 3, 4, 5, 1, 2).reshape(B, C * kh * kw, L)
        neg = 1.0 - pos

        # Concatenate to form all literals: [B, 2*C*kh*kw, L]
        literals = np.concatenate([pos, neg], axis=1)

        return {
            "literals": literals,                # [B, D, L] with D = 2*C*kh*kw
            "spatial": (H_out, W_out),
            "meta": {"C": C, "kh": kh, "kw": kw}
        }

# -----------------------------------------------------------------------------
# TMClauses
# -----------------------------------------------------------------------------
class TMClauses(nn.Module):
    r"""
    Clause definition + voting head in PyTorch.

    Inputs
    ------
    - literals: [B, D, L] in {0,1}, where D = 2*C*kh*kw (all positive+negative literals per patch).
    - We keep **fixed, discrete** clause definitions as binary selection masks over the D literals.
      Clauses are conjunctions (logical AND) across their selected literals, then OR across patches.

    Parameters
    ----------
    num_classes          : int
    clauses_per_class    : int (first half = positive, second half = negative)
    init_clause_mask     : Optional[torch.BoolTensor] of shape [M, D]
                           If None, starts with all-false (no literal included).
    learn_alpha          : bool (learn per-clause α by gradient descent)
    init_alpha           : float (initial α)

    Memory (dictionary)
    -------------------
    self.memory : {
        'literal_ta' : [M, D] int16   (placeholder for TA-like counters / your update rules)
        'fires'      : [M]   int32    (how many times each clause has fired)
        'uses'       : [M]   int32    (how many batch*patch opportunities were seen)
        'meta'       : dict           (bookkeeping)
    }
    """
    def __init__(
        self,
        num_classes: int,
        clauses_per_class: int,
        init_clause_mask: Optional[torch.Tensor] = None,
        learn_alpha: bool = True,
        init_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Store config; D (literals) will be inferred on first forward pass
        self.Cc = num_classes
        self.K  = clauses_per_class
        self.M  = num_classes * clauses_per_class
        self.D: Optional[int] = None  # To be initialized lazily
        self._init_clause_mask = init_clause_mask  # Store for lazy init
        self._device = device
        self._dtype = dtype
        self.clauses = {i+1:{"positive": [], "negative": []} for i in range(num_classes)}
        self.is_initialized = False

        # ----- Per-clause weights α (learnable, differentiable) -----
        if learn_alpha:
            self.alpha = nn.Parameter(torch.full((self.M,), float(init_alpha), dtype=dtype, device=device))
        else:
            self.register_buffer("alpha", torch.full((self.M,), float(init_alpha), dtype=dtype, device=device))

    def _lazy_initialize(self, D_literals: int):
        """Initializes buffers that depend on the number of literals, D."""
        print(f"TMClauses: Lazily initializing with D_literals = {D_literals}")
        self.D = D_literals
        device, dtype = self.alpha.device, self.alpha.dtype # Use device/dtype of existing parameters

        # ----- Clause definition mask (discrete): [M, D] bool -----
        if self._init_clause_mask is None:
            clause_mask = torch.zeros(self.M, self.D, dtype=torch.bool, device=device)
        else:
            assert self._init_clause_mask.shape == (self.M, self.D)
            clause_mask = self._init_clause_mask.to(device=device, dtype=torch.bool)

        # Store as a buffer (fixed, non-differentiable); update it manually with your feedback logic.
        self.register_buffer("clause_mask", clause_mask)

        # ----- Memory dictionary (TA counters, usage stats, etc.) -----
        self.memory: Dict[str, torch.Tensor] = {
            "literal_ta": torch.zeros(self.M, self.D, dtype=torch.int16, device=self.clause_mask.device),
            "fires":      torch.zeros(self.M, dtype=torch.int32, device=self.clause_mask.device),
            "uses":       torch.zeros(self.M, dtype=torch.int32, device=self.clause_mask.device),
            "meta":       {"num_classes": self.Cc, "clauses_per_class": self.K, "D_literals": self.D}
        }
        
        # Clean up and mark as initialized
        self.is_initialized = True
        del self._init_clause_mask

    def clause_composition(self, x_filters: np.ndarray, x_colors: np.ndarray, y: np.ndarray, literals_per_clause_proportion=0.05, batch_size=50, sampled_literal_trials=10):
        """
        x_filters: NumPy array of filter literals [B, D_filters, L]
        x_colors: NumPy array of color literals [B, D_colors, L]
        y: NumPy array of labels [B,]
        """
        # 1. Combine filter and color literals into a single NumPy array
        x_combined_np = np.concatenate([x_filters, x_colors], axis=1)
        one_hot_labels_np = self.one_hot_numpy(y)
        flattened_x = x_combined_np.reshape(x_combined_np.shape[0], x_combined_np.shape[1]* x_combined_np.shape[2])
        # Run logistic regression to analyze feature importance
        
        coefficients = self.run_logistic_regression(flattened_x, one_hot_labels_np, num_trials=50, num_features_to_sample=100)
        index = np.where(coefficients>1)
        print(coefficients.shape)
        print(len(index[0]), len(index[1]))
        input("Press enter to continue")
        # 2. Convert NumPy arrays to PyTorch tensors to work with the rest of the class
        device = self.alpha.device if self.is_initialized else 'cpu'
        x = torch.from_numpy(x_combined_np).to(device=device, dtype=self._dtype)
        y = torch.from_numpy(y).to(device=device)

        B, D, L = x.shape
        literals_per_clause_filters = 6
        literals_per_clause_colors = 2

        # Now, iterate through the data in mini-batches
        num_samples = B
        num_batches = (num_samples + batch_size - 1) // batch_size
        print(f"\nProcessing in {num_batches} mini-batches of size {batch_size}...")

        for i in range(num_batches):
            
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            labels_batch = y[start_idx:end_idx]
            one_hot_labels = self.one_hot(labels_batch)
            for k in range(sampled_literal_trials):
                sampled_indexes_filters = torch.randperm(x_filters.shape[1])[:literals_per_clause_filters]
                sampled_indexes_colors = torch.randperm(x_colors.shape[1])[:literals_per_clause_colors]+x_filters.shape[1]
                sampled_indexes = torch.cat([sampled_indexes_filters,sampled_indexes_colors])
                print(sampled_indexes)
                input('hipi')
                # Slice the literals tensor 'x' along the second dimension (D) using the sampled indices.
                sliced_literals = x[:, sampled_indexes, :]
                print(f"Shape of original literals: {x.shape}")
                print(f"Shape of sliced literals after feature selection: {sliced_literals.shape}")
                literals_batch = sliced_literals[start_idx:end_idx]
                or_patches = self.forward(literals_batch, clause_definition=True).int()
                mi_batch = []
                for j in range(one_hot_labels.shape[1]):
                    mi_batch.append(self.mutual_information(or_patches, one_hot_labels[:, j]))
                    
                mi_batch = torch.stack(mi_batch)
                best_column = mi_batch.argmax().item()
                if mi_batch[best_column].item()>1e-3:
                    positive = (one_hot_labels[:, best_column]*or_patches).sum()
                    negative = ((1-one_hot_labels[:, best_column])*(1-or_patches)).sum()
                    if positive.item()>negative.item():
                        self.evaluate_mutual_information_gain(best_column+1,sampled_indexes,"positive",x)
                        print("positive")
                    else:
                        self.evaluate_mutual_information_gain(best_column+1,sampled_indexes,"negative",x)
                        print("negative")
                input('hipi')
            print(f"  - Mini-batch {i+1} literals shape: {literals_batch.shape}, labels shape: {labels_batch.shape}")

    def evaluate_mutual_information_gain(self,clause_index,sampled_indexes,positive_or_negative,x,batch_size=50,threshold=0.8):
        existing_clauses = self.clauses[clause_index][positive_or_negative]
        sampled_rows = torch.randperm(x.shape[0])[:batch_size]
        sampled_x = x[sampled_rows]
        newly_evaluated_literals = sampled_x[:,sampled_indexes]
        or_patches = self.forward(newly_evaluated_literals, clause_definition=True)
        if len(existing_clauses)>0:
            for already_sampled_indexes in self.clauses[clause_index][positive_or_negative]:
                literal_batch_previously_evaluated = sampled_x[:,already_sampled_indexes]
                or_patches_previously_evaluated = self.forward(literal_batch_previously_evaluated, clause_definition=True)
                intersection = (or_patches & or_patches_previously_evaluated).sum()
                print(intersection)
                union = (or_patches | or_patches_previously_evaluated).sum()
                print(union)
                input('hipo')
                jaccard = intersection.float() / (union.float()+1e-6)
                if jaccard>threshold:
                    return
        else:
            self.clauses[clause_index][positive_or_negative].append(sampled_indexes)
        self.clauses[clause_index][positive_or_negative].append(sampled_indexes)

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor of class labels to a one-hot encoded tensor.

        Parameters:
        - y (torch.Tensor): A 1D or 2D tensor of class labels (integers). If 2D, it will be squeezed.

        Returns:
        - torch.Tensor: The one-hot encoded tensor of shape (len(y), self.Cc).
        """
        return F.one_hot(y.squeeze().long(), num_classes=self.Cc)

    def one_hot_numpy(self, y: np.ndarray) -> np.ndarray:
        """
        Converts a NumPy array of class labels to a one-hot encoded NumPy array.

        Parameters:
        - y (np.ndarray): A 1D or 2D array of class labels (integers). If 2D, it will be squeezed.

        Returns:
        - np.ndarray: The one-hot encoded array of shape (len(y), self.Cc), with dtype uint8.
        """
        y_squeezed = np.squeeze(y).astype(int)
        num_samples = y_squeezed.shape[0]
        one_hot_array = np.zeros((num_samples, self.Cc), dtype=np.uint8)
        one_hot_array[np.arange(num_samples), y_squeezed] = 1
        return one_hot_array

    def run_logistic_regression(self, flattened_x: np.ndarray, one_hot_labels: np.ndarray, num_trials: int = 10, num_features_to_sample: int = 100):
        """
        Performs multi-class logistic regression on random feature subsets to find important features.

        Parameters:
        - flattened_x (np.ndarray): The feature matrix of shape (num_samples, num_features).
        - one_hot_labels (np.ndarray): The one-hot encoded labels of shape (num_samples, num_classes).
        - num_trials (int): The number of times to run the regression on different feature subsets.
        - num_features_to_sample (int): The number of features to randomly sample in each trial.
        """
        print(f"\n--- Running Logistic Regression for feature analysis ({num_trials} trials) ---")
        num_samples, num_total_features = flattened_x.shape
        num_classes = one_hot_labels.shape[1]
        
        # Convert one-hot labels back to a 1D array of class indices for scikit-learn
        y_indices = np.argmax(one_hot_labels, axis=1)

        # This array will store the accumulated importance scores for each feature.
        aggregate_coeffs = np.zeros((num_classes, num_total_features))

        log_reg = LogisticRegression(multi_class='ovr', solver='lbfgs', C=0.1, max_iter=100, random_state=42)
        
        for i in range(num_trials):
            print(f"  - Trial {i+1}/{num_trials}...", end='\r')
            # Randomly sample 'num_features_to_sample' feature indices without replacement
            feature_indices = np.random.choice(num_total_features, size=num_features_to_sample, replace=False)
            
            # Create a view of the data with only the sampled features
            x_subset = flattened_x[:, feature_indices]
            
            try:
                log_reg.fit(x_subset, y_indices)
                # Calculate accuracy on the training subset
                accuracy = log_reg.score(x_subset, y_indices)
                # Calculate and display the 60th percentile of the absolute coefficient values
                abs_coeffs = np.abs(log_reg.coef_).mean(axis=0)
                quantile_60 = np.quantile(abs_coeffs, 0.60)
                print(f"  - Trial {i+1}/{num_trials}... Accuracy: {accuracy:.2f}, 60th percentile of |coeffs|: {quantile_60:.4f}", end='\r')
                relevant_features = np.where(abs_coeffs >= quantile_60)[0]
                # Add the coefficients from this trial back to the aggregate matrix at their original positions
                aggregate_coeffs[:, feature_indices[relevant_features]] += 1
            except Exception as e:
                print(f"\nAn error occurred during trial {i+1}: {e}")

        print(f"\nLogistic Regression analysis complete. Aggregate coefficients shape: {aggregate_coeffs.shape}")
        return aggregate_coeffs

    def mutual_information(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Calculates the mutual information I(X; Y) between two binary vectors.

        Parameters:
        - x (torch.Tensor): A 1D binary tensor.
        - y (torch.Tensor): A 1D binary tensor of the same length as x.
        - eps (float): A small epsilon to prevent log(0).

        Returns:
        - torch.Tensor: A scalar tensor representing the mutual information in bits.
        """
        x = x.float()
        y = y.float()

        # Probabilities for X
        p_x1 = x.mean()
        p_x0 = 1.0 - p_x1
        h_x = -(p_x0 * torch.log2(p_x0 + eps) + p_x1 * torch.log2(p_x1 + eps))

        # Probabilities for Y
        p_y1 = y.mean()
        p_y0 = 1.0 - p_y1
        h_y = -(p_y0 * torch.log2(p_y0 + eps) + p_y1 * torch.log2(p_y1 + eps))

        # Joint probabilities
        p_11 = (x * y).mean()
        p_01 = ((1 - x) * y).mean()
        p_10 = (x * (1 - y)).mean()
        p_00 = ((1 - x) * (1 - y)).mean()
        h_xy = -(p_00 * torch.log2(p_00 + eps) + p_01 * torch.log2(p_01 + eps) +
                 p_10 * torch.log2(p_10 + eps) + p_11 * torch.log2(p_11 + eps))

        # Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        return h_x + h_y - h_xy

    @torch.no_grad()
    def set_clause_mask(self, new_mask: torch.Tensor):
        """Replace the clause selection masks. new_mask: [M, D] bool."""
        assert new_mask.shape == (self.M, self.D)
        self.clause_mask.data.copy_(new_mask.to(dtype=torch.bool, device=self.clause_mask.device))

    @torch.no_grad()
    def ta_increment(self, delta: torch.Tensor):
        """
        Update TA-like counters in memory (external feedback logic can call this).
        delta: [M, D] int16 increments/decrements.
        """
        self.memory["literal_ta"].add_(delta.to(self.memory["literal_ta"].dtype))

    def forward(self, literals: torch.Tensor, clause_definition: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        literals: [B, D, L] in {0,1}  (from TMCNNClass)
        Returns:
            clause_map : [B, M, L] bool   (per-patch clause satisfaction; AND across selected literals)
            clause_or  : [B, M]   bool   (OR-pooled over patches)
            logits     : [B, Cc]  float  (weighted votes with per-clause α; pos - neg per class)
        """
        B, D, L = literals.shape

        # Lazily initialize buffers on the first forward pass
        if not self.is_initialized:
            self._lazy_initialize(D)

        assert D == self.D, f"Expected D={self.D}, got {D}."

        # --- Evaluate clauses on each patch ---
        # Mask selected literals -> [M, D] & [B, D, L] => gather as [B, M, L, Sel] via broadcasting.
        # Efficient trick: use masked_fill with -inf and take min == 1 (AND), but literals are {0,1},
        # so we can compute: for each clause m, for each patch ℓ, clause is satisfied iff
        #    min over selected literals equals 1
        # Build a selector to pick only the selected literals
        # literals_sel_mℓ = literals[:, selected_idx, ℓ]
        # We'll do this with boolean masking per clause in a vectorized way:

        # AND across selected literals by taking the product. Result is 1.0 iff all selected literals are 1.0.
        clause_map = literals.prod(dim=1).bool()  # [B,  L] bool
        # --- OR across patches (clause fires if any patch satisfies it) ---
        clause_or = clause_map.any(dim=-1)  # [B, 1] bool
        if not clause_definition:
            # --- Bookkeeping: update memory usage/fires counters (no grad) ---
            with torch.no_grad():
                self.memory["uses"]  += torch.tensor(L, device=clause_or.device, dtype=torch.int32)
                self.memory["fires"] += clause_or.sum(dim=0).to(self.memory["fires"].dtype)

            # --- Voting: per-class positive vs negative halves, weighted by α ---
            z = clause_or.float()                              # [B, M]
            alpha = self.alpha                                 # [M]
            scores = []
            for c in range(self.Cc):
                s, h = c*self.K, self.K//2
                pos = (z[:, s:s+h] * alpha[s:s+h]).sum(dim=1)
                neg = (z[:, s+h:s+self.K] * alpha[s+h:s+self.K]).sum(dim=1)
                scores.append(pos - neg)
            logits = torch.stack(scores, dim=1)               # [B, Cc]

            return clause_map, clause_or, logits
        else:
            return clause_or
