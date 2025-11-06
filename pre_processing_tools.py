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
        img : (H, W, 3) uint8 or float32
            RGB image. If uint8, values are assumed in [0,255]. If float, use your own scale.

        Returns
        -------
        self
        """
        assert img.ndim == 3 and img.shape[2] == 3, "img must be HxWx3 RGB."
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
        img_or_pixels : (H,W,3) or (N,3) array

        Returns
        -------
        labels : (H,W) or (N,) int
            Index of nearest centroid for each pixel.
        """
        X, shape = _to_pixels(img_or_pixels)
        C = self.centroids_
        dist2 = (
            np.sum(X**2, axis=1, keepdims=True)
            - 2 * (X @ C.T)
            + np.sum(C**2, axis=1, keepdims=True).T
        )
        labels = np.argmin(dist2, axis=1)
        return labels.reshape(shape[:2]) if len(shape) == 3 else labels

    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Quantize an RGB image using learned centroids.

        Parameters
        ----------
        img : (H,W,3) array

        Returns
        -------
        qimg : (H,W,3) same dtype as input
            Image with each pixel replaced by its centroid color.
        """
        labels = self.predict(img)
        C = self.centroids_
        q = C[labels.reshape(-1)].reshape(img.shape)
        # Preserve dtype if original was uint8
        if img.dtype == np.uint8:
            q = np.clip(np.rint(q), 0, 255).astype(np.uint8)
        else:
            q = q.astype(img.dtype)
        return q

def _to_pixels(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Utility: reshape HxWx3 or Nx3 into (N,3) and return original shape."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        H, W, _ = arr.shape
        X = arr.reshape(-1, 3).astype(np.float32)
        return X, (H, W, 3)
    elif arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float32), arr.shape
    else:
        raise ValueError("Expected shape (H,W,3) or (N,3).")



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


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    H, W = 128, 128
    img_gray = (np.random.rand(H, W) * 255).astype(np.float32)  # single-channel
    img_rgb  = (np.random.rand(H, W, 3) * 255).astype(np.float32)

    gs = [GaussianSpec(1.0), GaussianSpec(2.0)]
    ls = [LaplacianSpec(), LaplacianSpec(sigma=1.5)]
    thetas = [k * math.pi / 6 for k in range(6)]
    bs = [GaborSpec(theta=t, lambd=8.0) for t in thetas]

    bank = ClassicFilterBankNP(gs, ls, bs, per_channel=True, padding="reflect")

    out1 = bank.forward(img_gray)
    out2 = bank.forward(img_rgb)

    print(out1["all"].shape, out1["meta"])
    print(out2["all"].shape, out2["meta"])




# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    H, W = 128, 128
    img_gray = (np.random.rand(H, W) * 255).astype(np.float32)  # single-channel
    img_rgb  = (np.random.rand(H, W, 3) * 255).astype(np.float32)

    gs = [GaussianSpec(1.0), GaussianSpec(2.0)]
    ls = [LaplacianSpec(), LaplacianSpec(sigma=1.5)]
    thetas = [k * math.pi / 6 for k in range(6)]
    bs = [GaborSpec(theta=t, lambd=8.0) for t in thetas]
    
    bank = ClassicFilterBankNP(gs, ls, bs, per_channel=True, padding="reflect")
    # out1 = bank.forward(img_gray)
    out2 = bank.forward(img_rgb)
    print(out2['gabor'].shape)