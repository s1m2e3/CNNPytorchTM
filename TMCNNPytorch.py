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
        D_literals: int,
        init_clause_mask: Optional[torch.Tensor] = None,
        learn_alpha: bool = True,
        init_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.Cc = num_classes
        self.K  = clauses_per_class
        self.M  = num_classes * clauses_per_class
        self.D  = D_literals

        # ----- Clause definition mask (discrete): [M, D] bool -----
        if init_clause_mask is None:
            clause_mask = torch.zeros(self.M, self.D, dtype=torch.bool, device=device)
        else:
            assert init_clause_mask.shape == (self.M, self.D)
            clause_mask = init_clause_mask.to(device=device, dtype=torch.bool)

        # Store as a buffer (fixed, non-differentiable); update it manually with your feedback logic.
        self.register_buffer("clause_mask", clause_mask)

        # ----- Per-clause weights α (learnable, differentiable) -----
        if learn_alpha:
            self.alpha = nn.Parameter(torch.full((self.M,), float(init_alpha), dtype=dtype, device=device))
        else:
            self.register_buffer("alpha", torch.full((self.M,), float(init_alpha), dtype=dtype, device=device))

        # ----- Memory dictionary (TA counters, usage stats, etc.) -----
        self.memory: Dict[str, torch.Tensor] = {
            "literal_ta": torch.zeros(self.M, self.D, dtype=torch.int16, device=device),
            "fires":      torch.zeros(self.M, dtype=torch.int32, device=device),
            "uses":       torch.zeros(self.M, dtype=torch.int32, device=device),
            "meta":       {"num_classes": self.Cc, "clauses_per_class": self.K, "D_literals": self.D}
        }

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

    def forward(self, literals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        literals: [B, D, L] in {0,1}  (from TMCNNClass)
        Returns:
            clause_map : [B, M, L] bool   (per-patch clause satisfaction; AND across selected literals)
            clause_or  : [B, M]   bool   (OR-pooled over patches)
            logits     : [B, Cc]  float  (weighted votes with per-clause α; pos - neg per class)
        """
        B, D, L = literals.shape
        assert D == self.D, f"Expected D={self.D}, got {D}."

        # --- Evaluate clauses on each patch ---
        # Mask selected literals -> [M, D] & [B, D, L] => gather as [B, M, L, Sel] via broadcasting.
        # Efficient trick: use masked_fill with -inf and take min == 1 (AND), but literals are {0,1},
        # so we can compute: for each clause m, for each patch ℓ, clause is satisfied iff
        #    min over selected literals equals 1
        # Build a selector to pick only the selected literals
        # literals_sel_mℓ = literals[:, selected_idx, ℓ]
        # We'll do this with boolean masking per clause in a vectorized way:

        # Expand literals for broadcasting: [B, 1, D, L]
        lit = literals.unsqueeze(1)                 # [B, 1, D, L]
        mask = self.clause_mask.unsqueeze(0).unsqueeze(-1)  # [1, M, D, 1]
        # Keep only selected literals (others set to 1 so they don't affect AND/min)
        lit_masked = torch.where(mask, lit, torch.ones_like(lit))  # [B, M, D, L]
        # AND across selected literals == min across D equals 1  (since values are 0 or 1)
        clause_map = (lit_masked.min(dim=2).values == 1)  # [B, M, L] bool

        # --- OR across patches (clause fires if any patch satisfies it) ---
        clause_or = clause_map.any(dim=-1)  # [B, M] bool

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


# -----------------------------------------------------------------------------
# Minimal wiring example (CPU)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Fake binary batch: B=4, C=1, H=W=8
    B, C, H, W = 4, 1, 8, 8
    x = (torch.rand(B, C, H, W) > 0.7).float()

    # 1) Extract literals with TMCNNClass
    kh, kw = 3, 3
    extractor = TMCNNClass(in_channels=C, kernel_size=(kh, kw))
    out = extractor(x)
    literals = out["literals"]        # [B, 2*C*kh*kw, L]
    D = literals.shape[1]
    # # 2) Define TMClauses (e.g., 2 classes, 20 clauses/class)
    num_classes, K = 2, 20
    tm = TMClauses(num_classes, K, D_literals=D, learn_alpha=True, init_alpha=1.0)

    # # Optional: set an initial clause mask (here, randomly choose 5 literals per clause)
    # with torch.no_grad():
    #     init_mask = torch.zeros(tm.M, D, dtype=torch.bool)
    #     for m in range(tm.M):
    #         idx = torch.randperm(D)[:5]
    #         init_mask[m, idx] = True
    #     tm.set_clause_mask(init_mask)

    # # Forward: get per-patch clause map, per-sample clause OR, and class logits
    clause_map, clause_or, logits = tm(literals)
    print(clause_map.shape,clause_or.shape,logits.shape)
    # # logits can be trained with cross-entropy; only α is learnable by default
    # y = torch.randint(0, num_classes, (B,))
    # loss = F.cross_entropy(logits, y)
    # loss.backward()  # grads flow only to α
