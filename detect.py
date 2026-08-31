"""
detect.py
=========
Batch AI-generated image detector.

Takes a directory of images, scores each one, writes a JSON file.

    python detect.py --input ./images --output results.json

Output format:

    [
      {"image_path": "images/photo1.jpg", "pred": 0.8734, "label": "ai"},
      {"image_path": "images/photo2.jpg", "pred": 0.0121, "label": "real"}
    ]

`pred` is the confidence that the image is AI-generated:
0.0 = confidently real, 1.0 = confidently AI-generated.

`label` applies the operating threshold (default 0.35, selected on the
validation split) so the file is readable without knowing the cutoff.

Extra flags:
    --recursive        also walk subdirectories
    --per-model        include each branch's individual score
    --threshold 0.35   override the decision threshold
    --rf-weight 0.0    ensemble weight on the random forest (0 = CNN only)
    --models ./models  directory holding the weight files
    --batch-size 32    images per forward pass
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFile

from scoring import (
    Detector,
    DEFAULT_RF_WEIGHT,
    DEFAULT_THRESHOLD,
    pil_to_raw_tensor,
    normalize_for_cnn,
    extract_gradient_features,
    extract_fft_features,
)

# Some real-world files are truncated. Without this, one bad JPEG kills
# an entire run partway through.
ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root: str, recursive: bool = False) -> list[str]:
    """Return sorted image paths under root. Sorted so runs are reproducible."""
    paths = []
    if recursive:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in VALID_EXT:
                    paths.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in VALID_EXT:
                paths.append(p)
    return sorted(paths)


def score_batch(detector: Detector, paths: list[str], per_model: bool):
    """
    Score a list of image paths in one forward pass.

    Returns (results, failures). Batching matters: one-at-a-time on 10k
    images wastes most of the GPU.
    """
    tensors, kept, failures = [], [], []

    for p in paths:
        try:
            with Image.open(p) as img:
                tensors.append(pil_to_raw_tensor(img))
            kept.append(p)
        except Exception as e:
            failures.append((p, f"{type(e).__name__}: {e}"))

    if not kept:
        return [], failures

    raw = torch.cat(tensors).to(detector.device)
    results = []

    with torch.no_grad():
        # --- CNN branch (needs ImageNet normalization) ---
        logits = detector.cnn(normalize_for_cnn(raw)).squeeze(1)
        cnn_probs = torch.sigmoid(logits).cpu().numpy()

        # --- RF branch (needs raw [0,1], NO normalization) ---
        rf_probs = {}
        if detector.forests:
            g = extract_gradient_features(raw)
            f = extract_fft_features(raw)
            feats = torch.cat([g, f], dim=1).cpu().numpy().astype(np.float32)

            for name, rf in detector.forests.items():
                cols = feats[:, detector.feature_idx[name]]
                rf_probs[name] = rf.predict_proba(cols)[:, 1]

    for i, path in enumerate(kept):
        cnn_p = float(cnn_probs[i])

        combined = rf_probs.get("combined")
        if combined is not None and detector.rf_weight > 0:
            ens = (1.0 - detector.rf_weight) * cnn_p \
                  + detector.rf_weight * float(combined[i])
        else:
            ens = cnn_p

        row = {
            "image_path": path.replace("\\", "/"),
            "pred": round(ens, 6),
            "label": "ai" if ens > detector.threshold else "real",
        }

        if per_model:
            row["cnn"] = round(cnn_p, 6)
            for name, arr in rf_probs.items():
                row[name] = round(float(arr[i]), 6)

        results.append(row)

    return results, failures


def main():
    ap = argparse.ArgumentParser(
        description="Score images for AI-generated content."
    )
    ap.add_argument("--input", required=True,
                    help="directory containing images")
    ap.add_argument("--output", default="results.json",
                    help="path for the output JSON")
    ap.add_argument("--models", default="./models",
                    help="directory holding best_model.pth and rf_*.joblib")
    ap.add_argument("--recursive", action="store_true",
                    help="also search subdirectories")
    ap.add_argument("--per-model", action="store_true",
                    help="include each model's individual score")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="scores above this are labelled 'ai'")
    ap.add_argument("--rf-weight", type=float, default=DEFAULT_RF_WEIGHT,
                    help="ensemble weight on the random forest (0 = CNN only)")
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (auto-detected by default)")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"error: not a directory: {args.input}", file=sys.stderr)
        raise SystemExit(1)

    paths = find_images(args.input, args.recursive)
    if not paths:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        print(f"looked for: {', '.join(sorted(VALID_EXT))}", file=sys.stderr)
        raise SystemExit(1)

    print(f"found {len(paths)} images")
    print("loading models...")

    detector = Detector(
        model_dir=args.models,
        device=args.device,
        rf_weight=args.rf_weight,
        threshold=args.threshold,
    )

    print(f"device: {detector.device}")
    print(f"branches: cnn + {', '.join(detector.forests) or 'none'}")
    print(f"rf_weight: {detector.rf_weight} | threshold: {detector.threshold}\n")

    all_results, all_failures = [], []
    t0 = time.time()

    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i:i + args.batch_size]
        res, fails = score_batch(detector, chunk, args.per_model)
        all_results.extend(res)
        all_failures.extend(fails)

        done = min(i + args.batch_size, len(paths))
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  {done}/{len(paths)}  ({rate:.1f} img/s)", flush=True)

    with open(args.output, "w") as fh:
        json.dump(all_results, fh, indent=2)

    elapsed = time.time() - t0
    n_ai = sum(1 for r in all_results if r["label"] == "ai")
    n_real = len(all_results) - n_ai

    print(f"\nscored {len(all_results)} images in {elapsed:.1f}s")
    print(f"  AI-generated: {n_ai}")
    print(f"  real:         {n_real}")
    print(f"written to {args.output}")

    if all_failures:
        print(f"\n{len(all_failures)} file(s) could not be read:")
        for p, err in all_failures[:10]:
            print(f"  {p}: {err}")
        if len(all_failures) > 10:
            print(f"  ... and {len(all_failures) - 10} more")


if __name__ == "__main__":
    main()