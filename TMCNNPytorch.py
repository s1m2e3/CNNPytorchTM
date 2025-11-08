import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# -----------------------------------------------------------------------------
# TMCNNClass
# -----------------------------------------------------------------------------
class TMCNNClass(nn.Module):
    r"""
    Convolutional patch/literal extractor for Tsetlin-style pipelines.

    IMPORTANT: Inputs **must be binary** tensors in {0,1}.
               If you start from grayscale, binarize BEFORE calling forward.

    What it does
    ------------
    - Extracts all sliding patches of size (kh, kw) over C channels.
    - Produces *all possible literals* per patch:
        * positive literals  : x
        * negative literals  : 1 - x
      packed along the "literal" dimension so downstream modules can form clauses.

    Output
    ------
    Returns a dict with:
      - 'literals': [B, 2*C*kh*kw, L] float in {0,1}  (L = number of patches = H_out * W_out)
      - 'spatial': (H_out, W_out)                      (so you can map L back to 2-D)
      - 'meta': {'C': C, 'kh': kh, 'kw': kw}
    """
    def __init__(self, in_channels: int, kernel_size: Tuple[int, int]):
        super().__init__()
        self.C = in_channels
        self.kh, self.kw = kernel_size
        # We use Unfold for exact patch extraction (no learnable params).
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=1, padding=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert x.dtype in (torch.float32, torch.float16, torch.bfloat16) or x.dtype == torch.uint8, \
            "Input dtype should be float/uint8 with values in {0,1}."
        # x: [B, C, H, W] with values in {0,1}
        B, C, H, W = x.shape
        assert C == self.C, f"Expected {self.C} channels, got {C}."
        # Extract patches -> [B, C*kh*kw, L]
        pos = self.unfold(x.float())
        neg = 1.0 - pos
        # Concatenate all literals along feature (literal) dimension
        literals = torch.cat([pos, neg], dim=1)  # [B, 2*C*kh*kw, L]

        # Spatial size of the sliding windows
        H_out = H - self.kh + 1
        W_out = W - self.kw + 1

        return {
            "literals": literals,                # [B, D, L] with D = 2*C*kh*kw
            "spatial": (H_out, W_out),
            "meta": {"C": self.C, "kh": self.kh, "kw": self.kw}
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

    def clause_composition(self,x,y, literals_per_clause_proportion=0.05, batch_size=50, sampled_literal_trials=10):
        # Assuming x is a batch of literals of shape [B, D, L] where L is the number of patches 
        # Assuming y is a batch of labels of shape [B, C] where C is the number of classes
        B, D, L = x.shape
        literals_per_clause = 6
        

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
                sampled_indexes = torch.randperm(D)[:literals_per_clause]
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
