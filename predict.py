import os
from collections import Counter
from typing import Tuple

import torch
from PIL import Image

from constants.config import *
from model.crnn import CRNN
from ultis.common import ctc_decode, preprocess_image


# ------------------------------
# 📷 Stable Prediction on a Single Image
# ------------------------------
@torch.no_grad()
def predict_image(model: CRNN, image_path: str, runs: int = 3) -> Tuple[str, float]:
    """
    Run OCR prediction on a single image multiple times for stable results.

    Args:
        model (CRNN): The loaded OCR model
        image_path (str): Path to the image
        runs (int): Number of repeated predictions

    Returns:
        Tuple[str, float]: (Most common predicted text, Average confidence [0-1])
    """
    # Load and preprocess once (deterministic)
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)

    predictions = []
    confidences = []

    model.eval()  # ensure no dropout/batchnorm randomness
    torch.set_grad_enabled(False)
    for _ in range(runs):
        logits = model(image_tensor)  # (T, B, C)
        probs = torch.softmax(logits, dim=2)  # (T, B, C)
        max_probs, _ = torch.max(probs, dim=2)  # (T, B)

        # Decode with argmax-based CTC
        decoded = ctc_decode(logits, raw=False)
        pred_text, _ = decoded[0]

        predictions.append(pred_text)
        confidences.append(max_probs.mean().item())

    # Pick most common prediction across runs
    most_common_pred, _ = Counter(predictions).most_common(1)[0]
    avg_conf = sum(confidences) / len(confidences)

    return most_common_pred, avg_conf


# ------------------------------
# 🚀 Batch Prediction Entry Point
# ------------------------------
def main():
    # Load model
    model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    print(f"✅ Loaded model from: {BEST_MODEL_PATH}\n")

    # Load label mapping
    label_path = os.path.join(PREDICT_DIR, "labels.txt")
    if not os.path.exists(label_path):
        print(f"❌ labels.txt not found at: {label_path}")
        return

    label_map = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                fname, text = parts
                label_map[fname] = text.strip()

    # Image list
    images_dir = os.path.join(PREDICT_DIR, "images")
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return

    # Predict all
    for idx, fname in enumerate(image_files, start=1):
        fpath = os.path.join(images_dir, fname)

    pred_text, pred_conf, char_confs = predict_image(model, fpath, runs=3, return_char_conf=True)
    gt_text = label_map.get(fname, "")

    # Char-level accuracy
    total_chars = len(gt_text)
    correct_chars = sum(p == g for p, g in zip(pred_text, gt_text))
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0

    # Log results
    print(f"[{idx}] {fname}")
    print("-" * 40)
    print(f" 🔹 Ground Truth : {gt_text}")
    print(f" 🔸 Prediction   : {pred_text}")
    print(f" 🧠 Confidence   : {pred_conf:.2%}")
    print(f" 🎯 Char Accuracy: {char_acc:.2%}")
    print(f" 🔬 Char Confidences: {[f'{c:.2%}' for c in char_confs]}\n")


if __name__ == "__main__":
    main()
