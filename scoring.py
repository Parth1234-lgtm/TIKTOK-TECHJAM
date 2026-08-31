"""
scoring.py
==========
Shared scoring logic for AI-generated image detection.

Loads three models and produces a fake-probability for a single image:

  1. ConvNeXt-Tiny CNN          (best_model.pth)
  2. Random forest on gradient  (rf_gradient.joblib)   -- 14 Sobel features
  3. Random forest on FFT       (rf_fft.joblib)        --  9 spectral features
  4. Random forest on both      (rf_combined.joblib)   -- 23 features

Both detect.py (batch CLI) and the web app import from here, so the
preprocessing is guaranteed identical between them.

IMPORTANT: the two branches need DIFFERENT preprocessing.
  CNN  -> resize 224, to tensor, then ImageNet normalize
  RF   -> resize 224, to tensor, NO normalize (features expect [0,1])
Mixing these up produces garbage scores with no error message.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------
# Constants -- must match training exactly
# ---------------------------------------------------------------------

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_NAME = "convnext_tiny"

# Feature column order. The forests were trained on this exact ordering,
# so it must not change.
GRAD_COLS = [
    "mean_mag", "cov_xx", "cov_yy", "cov_xy", "trace", "det",
] + [f"mag_hist_{i}" for i in range(8)]

FFT_COLS = [f"radial_{i}" for i in range(8)] + ["hf_ratio"]
ALL_COLS = GRAD_COLS + FFT_COLS

GRAD_IDX = list(range(0, 14))
FFT_IDX = list(range(14, 23))
ALL_IDX = list(range(0, 23))

# Decision threshold chosen on the validation split (not on test).
# Accuracy was flat from 0.30-0.50; 0.35 balances real and fake error rates.
DEFAULT_THRESHOLD = 0.35

# Ensemble weight on the random forest. Set by sweeping on validation.
# 0.0 means CNN only. Update this once you have run the sweep.
DEFAULT_RF_WEIGHT = 0.2


# ---------------------------------------------------------------------
# Feature extraction (identical maths to the training-time versions)
# ---------------------------------------------------------------------

def _luma(images: torch.Tensor) -> torch.Tensor:
    """RGB -> grayscale luminance. images: (B,3,H,W) in [0,1]."""
    return (
        0.2126 * images[:, 0:1]
        + 0.7152 * images[:, 1:2]
        + 0.0722 * images[:, 2:3]
    )


def extract_gradient_features(images: torch.Tensor) -> torch.Tensor:
    """
    Sobel-gradient summary statistics.

    images: (B,3,H,W) float tensor in [0,1]
    returns: (B,14) tensor

    These describe the DISTRIBUTION of gradient values across the whole
    image, not where the edges are. Real photos carry signatures from lens
    blur, sensor noise and demosaicing that generated images lack.
    """
    B = images.shape[0]
    luma = _luma(images)

    sx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=images.device,
    ).view(1, 1, 3, 3) / 8.0

    sy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=images.device,
    ).view(1, 1, 3, 3) / 8.0

    gx = F.conv2d(luma, sx, padding=1)
    gy = F.conv2d(luma, sy, padding=1)

    mag = torch.sqrt(gx ** 2 + gy ** 2)
    mean_mag = mag.mean(dim=(1, 2, 3))

    gxf, gyf = gx.view(B, -1), gy.view(B, -1)
    gxc = gxf - gxf.mean(dim=1, keepdim=True)
    gyc = gyf - gyf.mean(dim=1, keepdim=True)

    cov_xx = (gxc * gxc).mean(dim=1)
    cov_yy = (gyc * gyc).mean(dim=1)
    cov_xy = (gxc * gyc).mean(dim=1)
    trace = cov_xx + cov_yy
    det = torch.clamp(cov_xx * cov_yy - cov_xy ** 2, min=0.0)

    hist = torch.stack(
        [torch.histc(mag[i], bins=8, min=0, max=1) for i in range(B)]
    )

    return torch.cat(
        [
            mean_mag[:, None], cov_xx[:, None], cov_yy[:, None],
            cov_xy[:, None], trace[:, None], det[:, None], hist,
        ],
        dim=1,
    )


# Radial masks depend only on image size, so build them once and reuse.
_MASK_CACHE: Dict[tuple, tuple] = {}


def _radial_masks(H: int, W: int, device, n_bins: int = 8):
    key = (H, W, str(device), n_bins)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    cy, cx = H // 2, W // 2
    yy = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, -1)
    radius = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = min(H, W) / 2.0

    rings = [
        ((radius >= (i / n_bins) * max_r) & (radius < ((i + 1) / n_bins) * max_r))
        for i in range(n_bins)
    ]
    hf = radius > max_r * 0.7
    lf = radius < max_r * 0.3

    _MASK_CACHE[key] = (rings, hf, lf)
    return rings, hf, lf


def extract_fft_features(images: torch.Tensor) -> torch.Tensor:
    """
    Frequency-domain summary from the 2D FFT magnitude spectrum.

    images: (B,3,H,W) float tensor in [0,1]
    returns: (B,9) tensor -- 8 radial energy bins + high/low frequency ratio

    Distance from the spectrum centre is frequency. Diffusion images tend to
    be too smooth (dim outer rings); GANs tend to oversharpen (bright outer
    rings). Real photos sit between, with sensor grain filling the highs.
    """
    gray = _luma(images)[:, 0]
    H, W = gray.shape[-2:]
    rings, hf, lf = _radial_masks(H, W, images.device)

    spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1)).abs()

    radial = torch.stack([spec[:, m].mean(dim=1) for m in rings], dim=1)
    hf_e = spec[:, hf].mean(dim=1)
    lf_e = spec[:, lf].mean(dim=1)
    ratio = (hf_e / (lf_e + 1e-10))[:, None]

    return torch.cat([radial, ratio], dim=1)


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def pil_to_raw_tensor(img: Image.Image, size: int = IMG_SIZE) -> torch.Tensor:
    """
    PIL image -> (1,3,size,size) float tensor in [0,1].

    This is the RF branch input. No normalization: the gradient and FFT
    feature functions assume pixel values in [0,1].
    """
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def normalize_for_cnn(raw: torch.Tensor) -> torch.Tensor:
    """
    (B,3,H,W) in [0,1] -> ImageNet-normalized, for the CNN branch.

    The pretrained ConvNeXt weights were trained on inputs normalized with
    these exact statistics. Skipping this silently degrades the CNN.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=raw.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=raw.device).view(1, 3, 1, 1)
    return (raw - mean) / std


# ---------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------

class Detector:
    """
    Wraps the CNN and the random forests behind one scoring call.

    Usage:
        det = Detector()
        scores = det.score_image("photo.jpg")
        # {'cnn': 0.87, 'gradient': 0.61, 'fft': 0.72,
        #  'combined': 0.69, 'ensemble': 0.87, 'label': 'ai'}
    """

    def __init__(
        self,
        model_dir: str = "./models",
        cnn_weights: str = "best_model.pth",
        device: Optional[str] = None,
        rf_weight: float = DEFAULT_RF_WEIGHT,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        import joblib
        import timm

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rf_weight = rf_weight
        self.threshold = threshold

        # ---- CNN ----
        cnn_path = os.path.join(model_dir, cnn_weights)
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"CNN weights not found: {cnn_path}")

        self.cnn = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)

        # map_location matters: the weights were saved from a GPU, and
        # without it this crashes on a CPU-only machine.
        state = torch.load(cnn_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.cnn.load_state_dict(state)
        self.cnn.to(self.device).eval()

        # ---- Random forests (all optional) ----
        self.forests: Dict[str, object] = {}
        for name, fname in [
            ("gradient", "rf_gradient.joblib"),
            ("fft", "rf_fft.joblib"),
            ("combined", "rf_combined.joblib"),
        ]:
            path = os.path.join(model_dir, fname)
            if os.path.exists(path):
                self.forests[name] = joblib.load(path)

        self.feature_idx = {
            "gradient": GRAD_IDX,
            "fft": FFT_IDX,
            "combined": ALL_IDX,
        }

    # -----------------------------------------------------------------

    def score_tensor(self, raw: torch.Tensor) -> Dict[str, float]:
        """
        raw: (1,3,H,W) float tensor in [0,1]
        returns dict of probabilities, each 0..1 where 1 means AI-generated.
        """
        raw = raw.to(self.device)
        out: Dict[str, float] = {}

        # --- CNN branch: needs normalized input ---
        with torch.no_grad():
            logit = self.cnn(normalize_for_cnn(raw)).squeeze(1)
            out["cnn"] = float(torch.sigmoid(logit).item())

        # --- RF branch: needs UNnormalized [0,1] input ---
        if self.forests:
            with torch.no_grad():
                g = extract_gradient_features(raw)
                f = extract_fft_features(raw)
                feats = torch.cat([g, f], dim=1).cpu().numpy().astype(np.float32)

            for name, rf in self.forests.items():
                cols = feats[:, self.feature_idx[name]]
                # predict_proba, not predict: we want the score, not 0/1.
                out[name] = float(rf.predict_proba(cols)[0, 1])

        # --- Weighted blend ---
        rf_score = out.get("combined")
        if rf_score is not None and self.rf_weight > 0:
            out["ensemble"] = (
                (1.0 - self.rf_weight) * out["cnn"] + self.rf_weight * rf_score
            )
        else:
            out["ensemble"] = out["cnn"]

        out["label"] = "ai" if out["ensemble"] > self.threshold else "real"
        return out

    def score_pil(self, img: Image.Image) -> Dict[str, float]:
        return self.score_tensor(pil_to_raw_tensor(img))

    def score_image(self, path: str) -> Dict[str, float]:
        with Image.open(path) as img:
            return self.score_pil(img)


# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python scoring.py <image-path>")
        raise SystemExit(1)

    det = Detector()
    result = det.score_image(sys.argv[1])
    for k, v in result.items():
        print(f"{k:>10}: {v if isinstance(v, str) else round(v, 4)}")
