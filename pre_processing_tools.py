import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Sequence
from joblib import Parallel, delayed, effective_n_jobs
import math
import cv2

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
    max_pixels_for_fit : Optional[int]
        If the number of pixels in the input image/batch exceeds this, a random sample of this size is used for fitting to save memory. If None, all pixels are used.
    random_state : Optional[int]
        Seed for reproducibility.
    n_jobs : Optional[int]
        Number of parallel jobs to run for fitting. -1 means using all available CPUs.

    Attributes (after fit)
    ----------------------
    centroids_ : (K, 3) float32
        RGB centroids in [0, 255] (if input is uint8) or input scale (if float).
    """
    def __init__(self, K: int, max_iters: int = 50, tol: float = 1e-3,
                 max_pixels_for_fit: Optional[int] = 1_000_000,
                 random_state: Optional[int] = None, n_jobs: Optional[int] = -1,
                 config: Optional[dict] = None):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.max_pixels_for_fit = max_pixels_for_fit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.centroids_: Optional[np.ndarray] = None

        # Load centroids from the config dictionary if they are provided
        if config and 'centroids' in config and config['centroids'] is not None:
            print("KMeansColorQuantizer: Loading centroids from config.")
            centroids_list = config['centroids']
            self.centroids_ = np.array(centroids_list, dtype=np.float32)
            # Verify shape
            if self.centroids_.shape != (self.K, 3):
                raise ValueError(f"Loaded centroids shape {self.centroids_.shape} does not match K={self.K}. Expected ({self.K}, 3).")
                
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
        if self.centroids_ is not None:
            print("KMeans.fit: Centroids already loaded from config. Skipping fitting.")
            return self
            
        print(f"KMeans.fit: Starting K-Means fitting with K={self.K}...")
        assert (img.ndim == 3 and img.shape[2] == 3) or \
               (img.ndim == 4 and img.shape[3] == 3), "img must be HxWx3 RGB or BxHxWx3 RGB."
        rng = np.random.default_rng(self.random_state)

        # --- Memory-efficient sampling for large datasets ---
        # Reshape to a 2D array of pixels (N_pixels, 3)
        all_pixels = img.reshape(-1, 3)
        N_total = all_pixels.shape[0]

        if self.max_pixels_for_fit and N_total > self.max_pixels_for_fit:
            print(f"KMeans: Input has {N_total:,} pixels, sampling {self.max_pixels_for_fit:,} for fitting.")
            sample_indices = rng.choice(N_total, self.max_pixels_for_fit, replace=False)
            X = all_pixels[sample_indices].astype(np.float32)
        else:
            X = all_pixels.astype(np.float32)
        N = X.shape[0]
        K = int(self.K)
        assert 1 <= K <= N, "K must be in [1, number of pixels]."

        # ---- K-Means++ initialization ----
        print("KMeans.fit: Starting K-Means++ initialization...")
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

        print("KMeans.fit: K-Means++ initialization complete.")
        C = np.stack(centers, axis=0)  # (K,3)

        # ---- Lloyd's iterations ----
        print("KMeans.fit: Starting Lloyd's iterations...")
        n_jobs = effective_n_jobs(self.n_jobs)
        print(f"KMeans.fit: Using {n_jobs} parallel jobs for Lloyd's iterations.")
        
        for i in range(self.max_iters):
            # Parallel E-step (assignment) and M-step (update)
            new_C, counts = _kmeans_lloyd_iter_parallel(X, C, n_jobs)

            # Check convergence
            shift = np.linalg.norm(new_C - C)
            C = new_C
            if shift <= self.tol:
                print(f"KMeans.fit: Converged after {i + 1} iterations.")
                break

        self.centroids_ = C.astype(np.float32)
        print(f"KMeans.fit: Fitting complete after {i + 1} iterations.")
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
        print("KMeans.predict: Starting prediction...")
        X, original_shape = _to_pixels(img_or_pixels)
        C = self.centroids_
        dist2 = (
            np.sum(X**2, axis=1, keepdims=True)
            - 2 * (X @ C.T)
            + np.sum(C**2, axis=1, keepdims=True).T
        )
        labels = np.argmin(dist2, axis=1)
        
        print("KMeans.predict: Prediction complete.")
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
        print("KMeans.transform: Starting color quantization...")
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
        print("KMeans.transform: Color quantization complete.")
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
        print("KMeans.get_distances: Starting distance calculation...")
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

        print("KMeans.get_distances: Distance calculation complete.")
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
# K-Means parallel helpers (top-level functions for joblib)
# --------------------------
def _lloyd_assignment_chunk(X_chunk: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    E-step for a chunk of data. Computes labels and sufficient statistics for the M-step.
    Returns labels for the chunk and the sum of points for each cluster.
    """
    K = C.shape[0]
    # dist2[i, k] = ||X[i] - C[k]||^2
    dist2 = (
        np.sum(X_chunk**2, axis=1, keepdims=True)
        - 2 * (X_chunk @ C.T)
        + np.sum(C**2, axis=1, keepdims=True).T
    )
    labels_chunk = np.argmin(dist2, axis=1)
    
    # Calculate sufficient statistics (sum of points and counts per cluster) for this chunk
    new_C_chunk = np.zeros_like(C, dtype=np.float64) # Use float64 for stable accumulation
    counts_chunk = np.bincount(labels_chunk, minlength=K)
    
    for k in range(K):
        if counts_chunk[k] > 0:
            new_C_chunk[k] += X_chunk[labels_chunk == k].sum(axis=0)
            
    return new_C_chunk, counts_chunk

def _kmeans_lloyd_iter_parallel(X: np.ndarray, C: np.ndarray, n_jobs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Performs one parallel iteration of Lloyd's algorithm (E-step and M-step)."""
    K = C.shape[0]
    
    # Parallel E-step and partial M-step
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_lloyd_assignment_chunk)(X_chunk, C)
        for X_chunk in np.array_split(X, n_jobs)
    )

    # Aggregate results from all chunks (M-step)
    new_C_sum = np.sum([res[0] for res in results], axis=0)
    total_counts = np.sum([res[1] for res in results], axis=0)

    # Finalize new centroids
    new_C = new_C_sum / np.maximum(total_counts[:, None], 1) # Avoid division by zero

    return new_C, total_counts


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
        batch_size: int = 32,
    ):
        print("ClassicFilterBankNP.__init__: Initializing filter bank...")
        self.dtype = dtype
        self.per_channel = per_channel
        self.padding = padding


        self.batch_size = batch_size
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
        print(f"ClassicFilterBankNP.__init__: Built {self.nG} Gaussian, {self.nL} Laplacian, {self.nB} Gabor kernels.")

    def _ensure_chw(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return (x_chw, C) as float32; x_chw is (C,H,W) or (B,C,H,W)."""
        assert x.ndim in (2, 3, 4), "x must be (H,W), (H,W,C), or (B,H,W,C)"
        if x.ndim == 2:
            x = x[None, :, :, None] # -> (1,H,W,1)
        elif x.ndim == 3:
            x = x[None, :, :, :] # -> (1,H,W,C)
        
        B, H, W, C = x.shape

        x = x.astype(self.dtype, copy=False)
        if not self.per_channel and C > 1:
            # convert to luminance-like by averaging channels
            x = np.mean(x, axis=2, keepdims=True)
            C = 1
        # to (C,H,W)
        return np.transpose(x, (0, 3, 1, 2)), C

    def _apply_family(self, x_chw: np.ndarray, kernels: List[np.ndarray]) -> np.ndarray:
        """Apply list of kernels to a single image (C,H,W) using OpenCV and concat results."""
        C, H, W = x_chw.shape # (Channels, Height, Width)
        outs = []

        # Map padding string to OpenCV's border types
        border_map = {
            "reflect": cv2.BORDER_REFLECT_101,
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "edge": cv2.BORDER_REPLICATE # Common fallback
        }
        border_type = border_map.get(self.padding, cv2.BORDER_REFLECT_101)

        for ker in kernels:
            # OpenCV's filter2D expects the image in (H, W, C) format.
            # We can apply the same kernel to all channels at once if C > 1.
            img_hwc = np.transpose(x_chw, (1, 2, 0)) # (H, W, C)
            
            # Apply convolution. -1 means output depth is same as input.
            filtered_img = cv2.filter2D(img_hwc, -1, ker, borderType=border_type)
            
            # Ensure output is (H, W, C) even if input was (H, W)
            if filtered_img.ndim == 2:
                filtered_img = filtered_img[:, :, np.newaxis]
            outs.append(filtered_img)

        if len(outs) == 0:
            return np.zeros((H, W, 0), dtype=self.dtype)
        
        # Concatenate all filter responses along the channel axis
        return np.concatenate(outs, axis=2).astype(self.dtype)

    def forward(self, x_bhwc: np.ndarray) -> Dict[str, np.ndarray]:
        print("\nClassicFilterBankNP.forward: Applying filter bank...")
        x_bchw, C_out = self._ensure_chw(x_bhwc)
        B, C, H, W = x_bchw.shape

        num_batches = (B + self.batch_size - 1) // self.batch_size
        all_results = []

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, B)
            batch_x = x_bchw[start_idx:end_idx]
            
            print(f"ClassicFilterBankNP.forward: Processing sub-batch {i+1}/{num_batches} (images {start_idx}-{end_idx-1})...")

            batch_results = []
            for j in range(batch_x.shape[0]): # Loop over images in the sub-batch
                x_chw_single = batch_x[j] # (C,H,W)
                
                # Apply filters to this single image
                G_responses = self._apply_family(x_chw_single, self.G_kers)
                L_responses = self._apply_family(x_chw_single, self.L_kers)
                B_responses = self._apply_family(x_chw_single, self.B_kers)
                
                ALL = np.concatenate([t for t in (G_responses, L_responses, B_responses) if t.shape[2] > 0], axis=2)
                batch_results.append(ALL)

            all_results.append(np.stack(batch_results, axis=0))

        # Concatenate results from all sub-batches
        final_responses = np.concatenate(all_results, axis=0)

        # The meta-information is consistent across the batch
        meta = {
            "n_gaussian": self.nG, "n_laplacian": self.nL, "n_gabor": self.nB,
            "per_channel": self.per_channel, "C_out": C_out,
            "input_shape": x_bhwc.shape
        }
        print("ClassicFilterBankNP.forward: Filter bank application complete.")
        # Note: This simplified version only returns 'all' and 'meta'.
        # You can expand it to return separated G, L, B if needed.
        return {"all": final_responses, "meta": meta}





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

    def __init__(self, cfg: Optional[BinarizerConfig] = None, config: Optional[dict] = None):
        self.cfg = cfg or BinarizerConfig()
        self.thr_filters_: Optional[np.ndarray] = None   # shape (F, 4) for (q20, q40, q60, q80)
        self.thr_kmeans_: Optional[np.ndarray] = None    # shape (K, 4)

        # Load thresholds from the config dictionary if they are provided
        if config:
            if 'thresholds_filters' in config and config['thresholds_filters'] is not None:
                print("Binarizer: Loading filter thresholds from config.")
                self.thr_filters_ = np.array(config['thresholds_filters'], dtype=np.float32)
            if 'thresholds_kmeans' in config and config['thresholds_kmeans'] is not None:
                print("Binarizer: Loading kmeans thresholds from config.")
                self.thr_kmeans_ = np.array(config['thresholds_kmeans'], dtype=np.float32)



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

    def fit_both(self, R: np.ndarray, D: np.ndarray) -> "Binarizer":
        """
        Fit thresholds on both filter responses (R) and KMeans distances (D).
        """
        self.fit_filters(R)
        self.fit_kmeans(D)
        return self

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
# Saving function with metadata
# --------------------------
def save_features_with_metadata_npz(feature_dict: dict, output_path: str, metadata: Optional[dict] = None):
    """
    Saves a nested dictionary of NumPy arrays to a compressed .npz file
    by flattening the dictionary keys, and includes a metadata dictionary.

    Parameters
    ----------
    feature_dict : dict
        The nested dictionary of feature arrays to save.
    output_path : str
        Path to the output .npz file.
    metadata : dict, optional
        A dictionary of metadata to save alongside the arrays.
    """
    # 1. Flatten the nested dictionary of arrays
    flat_dict = {}
    for source, source_dict in feature_dict.items():
        if source == 'filters':
            for filter_type, band_dict in source_dict.items():
                for band, data in band_dict.items():
                    key = f"{source}_{filter_type}_{band}"
                    flat_dict[key] = data
        elif source == 'kmeans':
            for band, data in source_dict.items():
                key = f"{source}_{band}"
                flat_dict[key] = data
        elif source == 'labels':
            # Directly add the labels array to the flat dictionary
            flat_dict['labels'] = source_dict
    # 2. Add metadata to the dictionary to be saved
    # We save the metadata dict as a 0-dimensional NumPy array of dtype=object.
    if metadata:
        # Use a key that is unlikely to clash with your feature keys
        flat_dict['_metadata'] = np.array(metadata, dtype=object)

    # 3. Save all items to a compressed .npz file
    np.savez_compressed(output_path, **flat_dict)
    print(f"\n--- Saving features with metadata ---")
    print(f"Features and metadata saved to {output_path}")


# --------------------------
# Main processing function
# --------------------------
def transform_batch(batch_img_rgb: np.ndarray, kmeans: KMeansColorQuantizer, bank: ClassicFilterBankNP, binarizer: Binarizer) -> dict:
    """
    Runs the transformation part of the preprocessing pipeline on a batch of images using pre-fitted processors.
    1. Applies a classic filter bank (Gaussian, Laplacian, Gabor).
    2. Computes K-Means color centroids and distances.
    3. Binarizes both feature sets into LOW, MID, HIGH bands.
    4. Organizes the final binary features into a structured dictionary.

    Parameters
    ----------
    batch_img_rgb : np.ndarray
        A batch of images with shape (B, H, W, 3) and dtype uint8.
    kmeans : KMeansColorQuantizer
        A pre-fitted K-Means quantizer.
    bank : ClassicFilterBankNP
        An initialized filter bank.
    binarizer : Binarizer
        A pre-fitted binarizer.

    Returns
    -------
    dict
        The final nested dictionary containing all binarized feature sets.
    """

    # Infer image dimensions from the input batch
    B, H, W, C_in = batch_img_rgb.shape
    
    # --- 1. K-Means distance calculation ---
    print(f"\nCalculating K-Means distances for batch with shape: {batch_img_rgb.shape}")
    distances = kmeans.get_distances(batch_img_rgb)
    print("Output distances shape (B, H, W, K):", distances.shape)

    # --- 2. Filter bank application ---
    filter_responses_batch = bank.forward(batch_img_rgb)["all"]
    print(f"Filter responses batch shape (B, H, W, F): {filter_responses_batch.shape}")

    # --- 3. Binarization (transform only) ---
    print("\nBinarizing features using pre-fitted thresholds...")
    binary_features, info = binarizer.transform_both(R=filter_responses_batch, D=distances)
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

def create_and_fit_processors(training_data: np.ndarray, config: dict) -> Tuple[KMeansColorQuantizer, ClassicFilterBankNP, Binarizer]:
    """
    Creates and fits the K-Means, Filter Bank, and Binarizer on the training data.

    Parameters
    ----------
    training_data : np.ndarray
        The training image data, shape (B, H, W, 3).
    config : dict
        The application configuration dictionary.

    Returns
    -------
    Tuple[KMeansColorQuantizer, ClassicFilterBankNP, Binarizer]
        A tuple containing the fitted K-Means model, the initialized filter bank,
        and the fitted Binarizer.
    """
    kmeans_cfg = config['kmeans']
    fb_cfg = config['filter_bank']
    binarizer_cfg = config.get('binarizer', {}) # Use .get for safe access

    # 1. --- Fit K-Means on training data ---
    print("\n--- Fitting K-Means on Training Data ---")
    # Initialize with config; if centroids are present, it will load them and skip fitting.
    kmeans = KMeansColorQuantizer(config=kmeans_cfg,
        K=kmeans_cfg['num_centroids'],
        random_state=kmeans_cfg['random_state'],
        max_pixels_for_fit=kmeans_cfg.get('max_pixels_for_fit')
    )
    kmeans.fit(training_data)
    print("K-Means fitting complete. Centroids:\n", kmeans.centroids_)

    # 2. --- Initialize Filter Bank ---
    print("\n--- Initializing Filter Bank ---")
    gs = [GaussianSpec(sigma=s) for s in fb_cfg['gaussian']['sigmas']]
    ls = [LaplacianSpec(sigma=s) for s in fb_cfg['laplacian']['sigmas']]
    thetas = np.linspace(0, math.pi, fb_cfg['gabor']['num_thetas'], endpoint=False)
    bs = [GaborSpec(theta=t, lambd=fb_cfg['gabor']['lambd']) for t in thetas]
    bank = ClassicFilterBankNP(
        gs, ls, bs,
        per_channel=fb_cfg['per_channel'],
        padding=fb_cfg['padding'],
        batch_size=fb_cfg.get('processing_batch_size', 32)
    )

    # 3. --- Fit Binarizer on a SAMPLE of training data features ---
    print("\n--- Fitting Binarizer on a Sample of Training Data Features ---")
    # Use a small, random sample of the training data to efficiently estimate quantile thresholds.
    # This avoids processing the entire training set just for fitting.
    sample_size = binarizer_cfg.get('fit_sample_size', 5000)
    print(f"Sampling {sample_size} images from training data to fit binarizer thresholds.")
    
    # Use a consistent random state for reproducibility
    rng = np.random.default_rng(kmeans_cfg.get('random_state'))
    sample_indices = rng.choice(training_data.shape[0], size=min(sample_size, training_data.shape[0]), replace=False)
    fit_sample_data = training_data[sample_indices]

    print("Generating features for the sample to learn binarization thresholds...")
    # Generate K-Means distances and Filter responses for this small sample
    training_kmeans_distances = kmeans.get_distances(fit_sample_data)
    training_filter_responses = bank.forward(fit_sample_data)["all"]

    # Initialize binarizer (it won't find thresholds in the config, so it will be unfitted)
    binarizer = Binarizer(config=config.get('binarizer'))
    # Now, fit it using the generated features
    binarizer.fit_both(R=training_filter_responses, D=training_kmeans_distances)
    print("Binarizer fitting complete.")

    return kmeans, bank, binarizer
