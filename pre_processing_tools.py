import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Sequence
import math

@dataclass
class KMeansColorQuantizer:
    """
    Simple K-Means(++) color quantizer for RGB images (no external deps).

    Parameters
    ----------
    K : int
        Number of color centroids (palette size).
    max_iters : int
        Maximum Lloyd iterations after K-Means++ initialization.
    tol : float
        Convergence tolerance on centroid movement (L2 norm).
    random_state : Optional[int]
        Seed for reproducibility.

    Attributes (after fit)
    ----------------------
    centroids_ : (K, 3) float32
        RGB centroids in [0, 255] (if input is uint8) or input scale (if float).
    """
    K: int
    max_iters: int = 50
    tol: float = 1e-3
    random_state: Optional[int] = None

    def fit(self, img: np.ndarray) -> "KMeansColorQuantizer":
        """
        Fit centroids to an RGB image.

        Parameters
        ----------
        img : (H, W, 3) or (B, H, W, 3) uint8 or float32
            RGB image(s). If uint8, values are assumed in [0,255]. If float, use your own scale.

        Returns
        -------
        self
        """
        assert (img.ndim == 3 and img.shape[2] == 3) or \
               (img.ndim == 4 and img.shape[3] == 3), "img must be HxWx3 RGB or BxHxWx3 RGB."
        rng = np.random.default_rng(self.random_state)

        X = img.reshape(-1, 3).astype(np.float32)  # (N,3)
        N = X.shape[0]
        K = int(self.K)
        assert 1 <= K <= N, "K must be in [1, number of pixels]."

        # ---- K-Means++ initialization ----
        # Choose first center uniformly
        idx0 = rng.integers(0, N)
        centers = [X[idx0]]

        # Squared distances to nearest chosen center
        d2 = np.sum((X - centers[0])**2, axis=1)

        for _ in range(1, K):
            probs = d2 / (d2.sum() + 1e-12)
            next_idx = rng.choice(N, p=probs)
            centers.append(X[next_idx])
            # update distances to nearest center
            new_d2 = np.sum((X - centers[-1])**2, axis=1)
            d2 = np.minimum(d2, new_d2)

        C = np.stack(centers, axis=0)  # (K,3)

        # ---- Lloyd's iterations ----
        for _ in range(self.max_iters):
            # Assignment step
            # Compute squared distances to each centroid in a vectorized way
            # dist2[i, k] = ||X[i] - C[k]||^2
            dist2 = (
                np.sum(X**2, axis=1, keepdims=True)
                - 2 * (X @ C.T)
                + np.sum(C**2, axis=1, keepdims=True).T
            )
            labels = np.argmin(dist2, axis=1)

            # Update step
            new_C = np.zeros_like(C)
            counts = np.bincount(labels, minlength=K).astype(np.float32)
            for k in range(K):
                if counts[k] > 0:
                    new_C[k] = X[labels == k].mean(axis=0)
                else:
                    # Re-seed empty cluster to a random point
                    new_C[k] = X[rng.integers(0, N)]

            # Check convergence
            shift = np.linalg.norm(new_C - C)
            C = new_C
            if shift <= self.tol:
                break

        self.centroids_ = C.astype(np.float32)
        return self

    def predict(self, img_or_pixels: np.ndarray) -> np.ndarray:
        """
        Assign each pixel to the nearest centroid.

        Parameters
        ----------
        img_or_pixels : (H,W,3), (B,H,W,3) or (N,3) array
        
        Returns
        -------
        labels : (H,W), (B,H,W) or (N,) int
            Index of nearest centroid for each pixel.
        """
        X, original_shape = _to_pixels(img_or_pixels)
        C = self.centroids_
        dist2 = (
            np.sum(X**2, axis=1, keepdims=True)
            - 2 * (X @ C.T)
            + np.sum(C**2, axis=1, keepdims=True).T
        )
        labels = np.argmin(dist2, axis=1)
        
        if len(original_shape) == 3: # (H,W,3) input
            return labels.reshape(original_shape[:2]) # (H,W)
        elif len(original_shape) == 4: # (B,H,W,3) input
            return labels.reshape(original_shape[:3]) # (B,H,W)
        else: # (N,3) input
            return labels # (N,)

    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Quantize an RGB image using learned centroids.

        Parameters
        ----------
        img : (H,W,3) or (B,H,W,3) array
        
        Returns
        -------
        qimg : (H,W,3) or (B,H,W,3) same dtype as input
            Image(s) with each pixel replaced by its centroid color.
        """
        labels = self.predict(img)
        C = self.centroids_
        
        if img.ndim == 3 or img.ndim == 4:
            q = C[labels.reshape(-1)].reshape(img.shape)
        else:
            raise ValueError("Input 'img' must be (H,W,3) or (B,H,W,3).")
        # Preserve dtype if original was uint8
        if img.dtype == np.uint8:
            q = np.clip(np.rint(q), 0, 255).astype(np.uint8)
        else:
            q = q.astype(img.dtype)
        return q

    def get_distances(self, img_or_pixels: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance from each pixel to each centroid.

        Parameters
        ----------
        img_or_pixels : (H,W,3) or (B,H,W,3) or (N,3) array

        Returns
        -------
        distances : (H,W,K) or (B,H,W,K) or (N,K) float32 array
            Euclidean distance from each pixel to each of the K centroids.
        """
        assert hasattr(self, "centroids_"), "KMeans must be fitted before calling get_distances."
        
        original_shape = img_or_pixels.shape
        if img_or_pixels.ndim not in [2, 3, 4]:
            raise ValueError("Input must be an image (H,W,3), a batch of images (B,H,W,3), or pixels (N,3).")

        X = img_or_pixels.reshape(-1, 3).astype(np.float32)
        C = self.centroids_

        # Compute squared Euclidean distances: (N, K)
        dist2 = (np.sum(X**2, axis=1, keepdims=True) - 2 * (X @ C.T) + np.sum(C**2, axis=1, keepdims=True).T)
        
        # Take the square root for Euclidean distance
        distances = np.sqrt(np.maximum(dist2, 0)) # Use np.maximum to avoid sqrt of small negative numbers

        return distances.reshape(*original_shape[:-1], self.K)

def _to_pixels(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Utility: reshape HxWx3 or Nx3 into (N,3) and return original shape."""
    if arr.ndim == 3 and arr.shape[2] == 3: # Single image (H,W,3)
        H, W, _ = arr.shape
        X = arr.reshape(-1, 3).astype(np.float32)
        return X, (H, W, 3)
    elif arr.ndim == 4 and arr.shape[3] == 3: # Batch of images (B,H,W,3)
        B, H, W, _ = arr.shape
        X = arr.reshape(-1, 3).astype(np.float32)
        return X, (B, H, W, 3)
    elif arr.ndim == 2 and arr.shape[1] == 3: # Already pixels (N,3)
        return arr.astype(np.float32), arr.shape
    else:
        raise ValueError("Expected shape (H,W,3), (B,H,W,3) or (N,3).")



# --------------------------
# Kernel builders (NumPy)
# --------------------------
def _odd_from_sigma(sigma: float, k: float = 6.0) -> int:
    """Kernel size ≈ k*sigma rounded up to nearest odd integer (>=3)."""
    size = max(3, int(math.ceil(k * sigma)))
    return size if size % 2 == 1 else size + 1

def gaussian_kernel2d(sigma: float, ksize: Optional[int] = None, dtype=np.float32) -> np.ndarray:
    assert sigma > 0
    k = _odd_from_sigma(sigma) if ksize is None else int(ksize)
    r = (k - 1) // 2
    y, x = np.mgrid[-r:r+1, -r:r+1].astype(dtype)
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(dtype)
    g /= g.sum() + 1e-12
    return g

def laplacian_kernel2d(dtype=np.float32) -> np.ndarray:
    return np.array([[0., 1., 0.],
                     [1., -4., 1.],
                     [0., 1., 0.]], dtype=dtype)

def log_kernel2d(sigma: float, ksize: Optional[int] = None, dtype=np.float32) -> np.ndarray:
    """Laplacian of Gaussian (LoG), zero-mean & scale-normalized."""
    assert sigma > 0
    k = _odd_from_sigma(sigma) if ksize is None else int(ksize)
    r = (k - 1) // 2
    y, x = np.mgrid[-r:r+1, -r:r+1].astype(dtype)
    r2 = (x**2 + y**2) / (2.0 * sigma**2)
    log = (r2 - 1.0) * np.exp(-r2)
    log = log - log.mean()
    log /= (sigma**2)
    return log.astype(dtype)

def gabor_kernel2d(
    theta: float,            # radians
    lambd: float,            # wavelength (pixels/cycle)
    sigma: Optional[float] = None,
    gamma: float = 0.5,      # aspect ratio (y/x)
    psi: float = 0.0,        # phase
    n_std: float = 3.0,
    dtype=np.float32
) -> np.ndarray:
    """Real Gabor kernel (cosine)."""
    assert lambd > 0
    sigma = (0.56 * lambd) if sigma is None else float(sigma)
    xmax = max(abs(n_std * sigma * math.cos(theta)), abs(n_std * sigma * gamma * math.sin(theta)))
    ymax = max(abs(n_std * sigma * math.sin(theta)), abs(n_std * sigma * gamma * math.cos(theta)))
    xmax = int(math.ceil(max(1.0, xmax)))
    ymax = int(math.ceil(max(1.0, ymax)))
    x = np.arange(-xmax, xmax + 1, dtype=dtype)
    y = np.arange(-ymax, ymax + 1, dtype=dtype)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    x_theta =  xx * math.cos(theta) + yy * math.sin(theta)
    y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
    gb = np.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / (sigma**2)) * np.cos(2.0 * math.pi * x_theta / lambd + psi)
    gb = gb - gb.mean()
    l1 = np.sum(np.abs(gb))
    if l1 > 0:
        gb = gb / l1
    return gb.astype(dtype)


# --------------------------
# Simple 2D convolution (NumPy, SAME padding)
# --------------------------
def conv2d_same(img: np.ndarray, ker: np.ndarray, padding: str = "reflect") -> np.ndarray:
    """
    img: HxW (single channel)
    ker: khxkw
    Returns SAME-sized result with chosen padding.
    """
    H, W = img.shape
    kh, kw = ker.shape
    ph, pw = (kh - 1) // 2, (kw - 1) // 2
    if padding == "reflect":
        pad_mode = "reflect"
        # For very small H,W vs kernel, fallback to edge to avoid reflect errors
        if H < kh or W < kw:
            pad_mode = "edge"
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode=pad_mode)
    elif padding == "constant":
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    else:
        raise ValueError("padding must be 'reflect' or 'constant'.")

    # Flip kernel for convolution
    kf = np.flipud(np.fliplr(ker))
    out = np.zeros_like(img, dtype=np.float32)

    # naive sliding (fine for prototyping; for speed use FFT or SciPy in practice)
    for i in range(H):
        ii = i + ph
        for j in range(W):
            jj = j + pw
            patch = padded[ii - ph: ii + ph + 1, jj - pw: jj + pw + 1]
            out[i, j] = np.sum(patch * kf, dtype=np.float32)
    return out


# --------------------------
# Specs + Filter bank (NumPy)
# --------------------------
@dataclass
class GaussianSpec:
    sigma: float
    ksize: Optional[int] = None  # explicit kernel size (odd)

@dataclass
class LaplacianSpec:
    sigma: Optional[float] = None  # None => 3x3 Laplacian; else LoG(sigma)
    ksize: Optional[int] = None

@dataclass
class GaborSpec:
    theta: float
    lambd: float
    sigma: Optional[float] = None
    gamma: float = 0.5
    psi: float = 0.0
    n_std: float = 3.0


class ClassicFilterBankNP:
    """
    NumPy-only classic filter bank: Gaussian, Laplacian/LoG, and Gabor filters.

    Input
    -----
    - x: ndarray with shape (H, W) for single-channel or (H, W, C) for multi-channel.
         If you pass color, filters are applied independently to each channel and concatenated along channel axis.

    Configuration (user-specified)
    ------------------------------
    - gaussian_specs  : list[GaussianSpec], e.g., [GaussianSpec(1.0), GaussianSpec(2.0)]
    - laplacian_specs : list[LaplacianSpec], e.g., [LaplacianSpec(), LaplacianSpec(sigma=1.5)]
    - gabor_specs     : list[GaborSpec],    e.g., [GaborSpec(theta=0, lambd=8), ...]
    - per_channel     : if True, apply each filter to each input channel independently;
                        if False and x has C>1, we first average channels to luminance before filtering.

    Output
    ------
    dict with:
      - 'gaussian' : (H, W, nG*C_out)
      - 'laplacian': (H, W, nL*C_out)
      - 'gabor'    : (H, W, nB*C_out)
      - 'all'      : (H, W, (nG+nL+nB)*C_out)
      - 'meta'     : info dict
    """
    def __init__(
        self,
        gaussian_specs: Sequence[GaussianSpec],
        laplacian_specs: Sequence[LaplacianSpec],
        gabor_specs: Sequence[GaborSpec],
        per_channel: bool = True,
        dtype: np.dtype = np.float32,
        padding: str = "reflect",
    ):
        self.dtype = dtype
        self.per_channel = per_channel
        self.padding = padding

        # Build kernels
        self.G_kers: List[np.ndarray] = [gaussian_kernel2d(s.sigma, s.ksize, dtype=dtype) for s in gaussian_specs]
        self.L_kers: List[np.ndarray] = []
        for s in laplacian_specs:
            if s.sigma is None:
                self.L_kers.append(laplacian_kernel2d(dtype=dtype))
            else:
                self.L_kers.append(log_kernel2d(s.sigma, s.ksize, dtype=dtype))
        self.B_kers: List[np.ndarray] = [gabor_kernel2d(s.theta, s.lambd, s.sigma, s.gamma, s.psi, s.n_std, dtype=dtype)
                                         for s in gabor_specs]

        self.nG, self.nL, self.nB = len(self.G_kers), len(self.L_kers), len(self.B_kers)

    def _ensure_chw(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return (x_chw, C) as float32; x_chw is (C,H,W)."""
        assert x.ndim in (2, 3), "x must be (H,W) or (H,W,C)"
        if x.ndim == 2:
            x = x[:, :, None]
        H, W, C = x.shape
        x = x.astype(self.dtype, copy=False)
        if not self.per_channel and C > 1:
            # convert to luminance-like by averaging channels
            x = np.mean(x, axis=2, keepdims=True)
            C = 1
        # to (C,H,W)
        return np.transpose(x, (2, 0, 1)), C

    def _apply_family(self, x_chw: np.ndarray, kernels: List[np.ndarray]) -> np.ndarray:
        """Apply list of kernels to (C,H,W) and concat along channel -> (H,W, len*k_out)."""
        C, H, W = x_chw.shape
        outs = []
        for ker in kernels:
            if self.per_channel:
                # depthwise: one output per input channel
                for c in range(C):
                    outs.append(conv2d_same(x_chw[c], ker, padding=self.padding)[:, :, None])
            else:
                # shared filter on the (single) channel
                outs.append(conv2d_same(x_chw[0], ker, padding=self.padding)[:, :, None])
        if len(outs) == 0:
            return np.zeros((H, W, 0), dtype=self.dtype)
        return np.concatenate(outs, axis=2)

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        x_chw, C_out = self._ensure_chw(x)
        G = self._apply_family(x_chw, self.G_kers)  # (H,W,nG*C_out)
        L = self._apply_family(x_chw, self.L_kers)  # (H,W,nL*C_out)
        B = self._apply_family(x_chw, self.B_kers)  # (H,W,nB*C_out)
        ALL = np.concatenate([t for t in (G, L, B) if t.shape[2] > 0], axis=2) if (self.nG + self.nL + self.nB) > 0 \
              else np.zeros((x_chw.shape[1], x_chw.shape[2], 0), dtype=self.dtype)
        meta = {
            "n_gaussian": self.nG, "n_laplacian": self.nL, "n_gabor": self.nB,
            "per_channel": self.per_channel, "C_out": C_out
        }
        return {"gaussian": G, "laplacian": L, "gabor": B, "all": ALL, "meta": meta}





@dataclass
class BinarizerConfig:
    # Overlap quantiles (inclusive ranges)
    low_hi: float = 0.40   # LOW:   (-inf, q_low_hi]
    mid_lo: float = 0.20   # MID:   [q_mid_lo, q_mid_hi]
    mid_hi: float = 0.80
    high_lo: float = 0.60  # HIGH:  [q_high_lo, +inf)

class Binarizer:
    """
    Quantile-based, overlapping binarizer for:
      1) Filter responses R ∈ ℝ^{B×H×W×F}
      2) KMeans distances D ∈ ℝ^{B×H×W×K}

    Bands per channel (over batch+spatial axes):
      LOW  : x ≤ q_low_hi
      MID  : q_mid_lo ≤ x ≤ q_mid_hi
      HIGH : x ≥ q_high_lo
    with default overlaps: LOW(0–40), MID(20–80), HIGH(60–100) percentiles.

    Notes
    -----
    - Thresholds are computed **per channel** and cached in `self.thr_filters_` / `self.thr_kmeans_`.
    - Outputs are uint8 in {0,1}.
    - You can fit on one dataset (train) and transform on another (val/test).
    """

    def __init__(self, cfg: Optional[BinarizerConfig] = None):
        self.cfg = cfg or BinarizerConfig()
        self.thr_filters_: Optional[np.ndarray] = None   # shape (F, 4) for (q20, q40, q60, q80)
        self.thr_kmeans_: Optional[np.ndarray] = None    # shape (K, 4)

    # -------------------------
    # Filters: fit + transform
    # -------------------------
    def fit_filters(self, R: np.ndarray) -> "Binarizer":
        """
        Compute per-filter thresholds from responses R ∈ ℝ^{B×H×W×F}.
        Stores thresholds in self.thr_filters_ as (F,4): [q20,q40,q60,q80].
        """
        assert R.ndim == 4, "R must be [B,H,W,F]"
        B, H, W, F = R.shape
        flat = R.reshape(B*H*W, F).astype(np.float32)

        q20 = np.quantile(flat, self.cfg.mid_lo, axis=0)
        q40 = np.quantile(flat, self.cfg.low_hi, axis=0)
        q60 = np.quantile(flat, self.cfg.high_lo, axis=0)
        q80 = np.quantile(flat, self.cfg.mid_hi, axis=0)

        self.thr_filters_ = np.stack([q20, q40, q60, q80], axis=0).T  # (F,4)
        return self

    def transform_filters(self, R: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply stored thresholds to responses R ∈ ℝ^{B×H×W×F}.
        Returns:
          Bf : uint8 binaries ∈ {0,1}^{B×H×W×(3F)} in channel order [LOW(0..F-1), MID, HIGH]
          info : dict with 'thr_filters' ((F,4) array)
        """
        assert self.thr_filters_ is not None, "Call fit_filters first."
        assert R.ndim == 4, "R must be [B,H,W,F]"
        B, H, W, F = R.shape
        q20, q40, q60, q80 = self.thr_filters_.T  # (F,)

        # Broadcast thresholds to [B,H,W,F]
        Rf = R.astype(np.float32)
        LOW  = (Rf <= q40[None, None, None, :]).astype(np.uint8)
        MID  = ((Rf >= q20[None, None, None, :]) & (Rf <= q80[None, None, None, :])).astype(np.uint8)
        HIGH = (Rf >= q60[None, None, None, :]).astype(np.uint8)

        Bf = np.concatenate([LOW, MID, HIGH], axis=3)  # (B,H,W,3F)
        info = {"thr_filters": self.thr_filters_.copy()}
        return Bf, info

    def fit_transform_filters(self, R: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self.fit_filters(R)
        return self.transform_filters(R)

    # -------------------------
    # KMeans distances: fit + transform
    # -------------------------
    def fit_kmeans(self, D: np.ndarray) -> "Binarizer":
        """
        Compute per-centroid thresholds from distances D ∈ ℝ^{B×H×W×K}.
        Stores thresholds in self.thr_kmeans_ as (K,4): [q20,q40,q60,q80].
        """
        assert D.ndim == 4, "D must be [B,H,W,K]"
        B, H, W, K = D.shape
        flat = D.reshape(B*H*W, K).astype(np.float32)

        q20 = np.quantile(flat, self.cfg.mid_lo, axis=0)
        q40 = np.quantile(flat, self.cfg.low_hi, axis=0)
        q60 = np.quantile(flat, self.cfg.high_lo, axis=0)
        q80 = np.quantile(flat, self.cfg.mid_hi, axis=0)

        self.thr_kmeans_ = np.stack([q20, q40, q60, q80], axis=0).T  # (K,4)
        return self

    def transform_kmeans(self, D: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply stored thresholds to distances D ∈ ℝ^{B×H×W×K}.
        Returns:
          Bd : uint8 binaries ∈ {0,1}^{B×H×W×(3K)} with channel order [LOW(close), MID, HIGH(far)]
          info : dict with 'thr_kmeans' ((K,4) array)
        """
        assert self.thr_kmeans_ is not None, "Call fit_kmeans first."
        assert D.ndim == 4, "D must be [B,H,W,K]"
        B, H, W, K = D.shape
        q20, q40, q60, q80 = self.thr_kmeans_.T

        Df = D.astype(np.float32)
        # For distances: LOW=close (≤q40); MID=q20..q80; HIGH=far (≥q60)
        LOW  = (Df <= q40[None, None, None, :]).astype(np.uint8)
        MID  = ((Df >= q20[None, None, None, :]) & (Df <= q80[None, None, None, :])).astype(np.uint8)
        HIGH = (Df >= q60[None, None, None, :]).astype(np.uint8)

        Bd = np.concatenate([LOW, MID, HIGH], axis=3)  # (B,H,W,3K)
        info = {"thr_kmeans": self.thr_kmeans_.copy()}
        return Bd, info

    def fit_transform_kmeans(self, D: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self.fit_kmeans(D)
        return self.transform_kmeans(D)

    # -------------------------
    # Convenience: combine both
    # -------------------------
    def fit_transform_both(self, R: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Fit thresholds on both inputs, return concatenated binaries:
          Ball ∈ {0,1}^{B×H×W×(3F+3K)} and info dict with both threshold tables.
        """
        Bf, info_f = self.fit_transform_filters(R)
        Bd, info_d = self.fit_transform_kmeans(D)
        Ball = np.concatenate([Bf, Bd], axis=3)
        info = {**info_f, **info_d}
        return Ball, info

    def transform_both(self, R: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Transform with already-fitted thresholds; returns concatenated binaries and threshold tables.
        """
        Bf, info_f = self.transform_filters(R)
        Bd, info_d = self.transform_kmeans(D)
        Ball = np.concatenate([Bf, Bd], axis=3)
        info = {**info_f, **info_d}
        return Ball, info
# --------------------------
# Main processing function
# --------------------------
def process_batch(batch_img_rgb: np.ndarray, config: dict) -> dict:
    """
    Runs the full preprocessing pipeline on a batch of images.
    1. Applies a classic filter bank (Gaussian, Laplacian, Gabor).
    2. Computes K-Means color centroids and distances.
    3. Binarizes both feature sets into LOW, MID, HIGH bands.
    4. Organizes the final binary features into a structured dictionary.

    Parameters
    ----------
    batch_img_rgb : np.ndarray
        A batch of images with shape (B, H, W, 3) and dtype uint8.
    config : dict
        A dictionary loaded from config.yaml containing all processing parameters.

    Returns
    -------
    dict
        The final nested dictionary containing all binarized feature sets.
    """
    # --- Setup parameters from config ---
    kmeans_cfg = config['kmeans']
    fb_cfg = config['filter_bank']
    
    # Infer image dimensions from the input batch
    B, H, W, C_in = batch_img_rgb.shape

    # Build filter specs from config
    gs = [GaussianSpec(sigma=s) for s in fb_cfg['gaussian']['sigmas']]
    ls = [LaplacianSpec(sigma=s) for s in fb_cfg['laplacian']['sigmas']]
    thetas = np.linspace(0, math.pi, fb_cfg['gabor']['num_thetas'], endpoint=False)
    bs = [GaborSpec(theta=t, lambd=fb_cfg['gabor']['lambd']) for t in thetas]

    # --- Create dummy data and filter bank ---
    bank = ClassicFilterBankNP(gs, ls, bs, per_channel=fb_cfg['per_channel'], padding=fb_cfg['padding'])
    
    # --- K-Means quantization example ---
    print("\n--- K-Means Example ---")
    kmeans = KMeansColorQuantizer(K=kmeans_cfg['num_centroids'], random_state=kmeans_cfg['random_state'])
    
    print(f"Fitting K-Means with {kmeans.K} centroids on the RGB image...")
    kmeans.fit(batch_img_rgb)
    quantized_img = kmeans.transform(batch_img_rgb)
    print("K-Means centroids (RGB):\n", kmeans.centroids_)
    print("Quantized image shape:", quantized_img.shape, "and dtype:", quantized_img.dtype)

    # --- K-Means distance calculation example ---
    print("\n--- K-Means Distance Calculation Example ---")
    print(f"Calculating distances for a batch of images with shape: {batch_img_rgb.shape}")
    distances = kmeans.get_distances(batch_img_rgb)
    print("Output distances shape (B, H, W, K):", distances.shape)

    # --- Binarization example ---
    print("\n--- Binarization Example ---")
    # 1. Apply the filter bank to the same batch of images to get filter responses.
    #    We loop because the current filter bank processes one image at a time.
    filter_responses_batch = np.stack(
        [bank.forward(img)["all"] for img in batch_img_rgb],
        axis=0
    )
    print(f"Filter responses batch shape (B, H, W, F): {filter_responses_batch.shape}")

    # 2. Instantiate the Binarizer and binarize both filter responses and distances.
    binarizer = Binarizer()
    binary_features, info = binarizer.fit_transform_both(R=filter_responses_batch, D=distances)
    print(f"Combined binary features shape (B, H, W, 3F+3K): {binary_features.shape}")

    # --- Organize binarized features into a structured dictionary ---
    print("\n--- Organizing features into a dictionary ---")
    F = filter_responses_batch.shape[-1]
    K = distances.shape[-1]

    # 1. First, separate the main LOW, MID, HIGH blocks for filters and kmeans
    low_filters  = binary_features[:, :, :, 0:F]
    mid_filters  = binary_features[:, :, :, F:2*F]
    high_filters = binary_features[:, :, :, 2*F:3*F]
    low_kmeans   = binary_features[:, :, :, 3*F:3*F+K]
    mid_kmeans   = binary_features[:, :, :, 3*F+K:3*F+2*K]
    high_kmeans  = binary_features[:, :, :, 3*F+2*K:]

    # 2. Get the number of channels for each filter family (accounting for per_channel=True)
    n_gauss_ch = bank.nG * C_in
    n_lapl_ch  = bank.nL * C_in
    n_gabor_ch = bank.nB * C_in

    # 3. Define the slice boundaries for filter families
    gauss_end = n_gauss_ch
    lapl_end  = gauss_end + n_lapl_ch

    # 4. Build the nested dictionary
    feature_dict = {
        'filters': {
            'gaussian': {
                'low':  low_filters[:, :, :, :gauss_end],
                'mid':  mid_filters[:, :, :, :gauss_end],
                'high': high_filters[:, :, :, :gauss_end]
            },
            'laplacian': {
                'low':  low_filters[:, :, :, gauss_end:lapl_end],
                'mid':  mid_filters[:, :, :, gauss_end:lapl_end],
                'high': high_filters[:, :, :, gauss_end:lapl_end]
            },
            'gabor': {
                'low':  low_filters[:, :, :, lapl_end:],
                'mid':  mid_filters[:, :, :, lapl_end:],
                'high': high_filters[:, :, :, lapl_end:]
            }
        },
        'kmeans': {
            'low':  low_kmeans,
            'mid':  mid_kmeans,
            'high': high_kmeans
        }
    }

    # 5. Print all shapes inside the dictionary to verify the structure
    print("\n--- Verifying all shapes in the dictionary ---")
    for source, source_dict in feature_dict.items():
        if source == 'filters':
            for filter_type, band_dict in source_dict.items():
                for band, data in band_dict.items():
                    print(f"Shape of ['{source}']['{filter_type}']['{band}']: {data.shape}")
        elif source == 'kmeans':
            for band, data in source_dict.items():
                print(f"Shape of ['{source}']['{band}']: {data.shape}")
    
    return feature_dict


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

    # --- Create a dummy batch of images for demonstration ---
    # For demonstration, we define dummy dimensions here. The process_batch function infers them from its input.
    dummy_batch = (np.random.rand(4, 128, 128, 3) * 255).astype(np.uint8) # Example with batch size=4, H=W=128

    # --- Run the processing pipeline ---
    final_features = process_batch(dummy_batch, config)
    print("\nProcessing complete. The final feature dictionary has been generated.")