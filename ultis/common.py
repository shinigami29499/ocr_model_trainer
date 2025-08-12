from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from constants.config import *


# ------------------------------
# 🔤 Text Encoding Utilities
# ------------------------------
def text_to_labels(text: str) -> List[int]:
    """Convert text string to list of character indices."""
    return [CHAR_TO_INDEX[c] for c in text if c in CHAR_TO_INDEX]


def labels_to_text(label: List[int]) -> str:
    """Convert list of character indices to text string (skip CTC blank)."""
    return ''.join([INDEX_TO_CHAR.get(i, '') for i in label if i != 0])


# ------------------------------
# 🧼 Image Preprocessing Function
# ------------------------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for CRNN OCR model.
    Ensures identical steps for training & prediction.
    
    Args:
        image (PIL.Image): Input image.

    Returns:
        torch.Tensor: Normalized tensor of shape (1, H, W).
    """
    # Convert to grayscale
    image = image.convert("L")

    # Get original width and height
    w, h = image.size

    # No resize — keep original size

    # If width < TARGET_WIDTH, pad right side; else crop or keep as is
    pad_width = max(0, TARGET_WIDTH - w)
    image = TF.pad(image, (0, 0, pad_width, 0), fill=255)  # white padding on right

    # Optionally, if width > TARGET_WIDTH, crop to TARGET_WIDTH
    if image.width > TARGET_WIDTH:
        image = image.crop((0, 0, TARGET_WIDTH, image.height))

    # To tensor & normalize [-1, 1]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(image)  # (1, H, W)

    return tensor



# ----------------------------------------
# 🔧 collate_fn: Prepares OCR batch
# ----------------------------------------
def collate_fn(
    batch: List[Tuple[torch.Tensor, str]],
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    """
    Pads images to same width and encodes text labels to integer indices.
    """
    images, texts = zip(*batch)

    # Pad all images to the same width
    max_width = max([img.shape[2] for img in images])
    images_padded = [
        torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, 0), value=0.0)
        for img in images
    ]
    images_padded = torch.stack(images_padded)  # (B, C, H, W)

    # Encode all text labels
    targets = [encode_text(text) for text in texts]
    lengths = [len(t) for t in targets]

    return images_padded, targets, lengths


# ----------------------------------------
# 🔡 encode_text: Converts text to indices
# ----------------------------------------
def encode_text(text: str) -> torch.Tensor:
    """
    Converts a string label into a tensor of character indices using CHARSET.
    Used to convert ground truth text into training targets for CTC loss.

    Args:
        text (str): Text label (e.g., "Ab12")

    Returns:
        torch.Tensor: Encoded tensor of indices (e.g., [26, 1, 52, 53])
    """
    return torch.tensor([CHAR_TO_INDEX[c] for c in text], dtype=torch.long)


# ----------------------------------------
# 🔁 decode_text: Converts indices to text
# ----------------------------------------
def decode_text(indices: List[int], raw: bool = False) -> str:
    """
    Converts model-predicted indices back into human-readable text.

    If raw=False (default), applies CTC decoding rules:
        - Removes repeated characters
        - Removes blank tokens (CTC blank = len(CHARSET))

    If raw=True, returns the raw character sequence.

    Args:
        indices (List[int]): Sequence of predicted indices
        raw (bool): Whether to skip CTC decoding (default: False)

    Returns:
        str: Decoded string (e.g., [0, 0, 1] -> "ab")
    """
    decoded = []
    prev = BLANK_IDX

    for idx in indices:
        if raw or (idx != prev and idx != BLANK_IDX):
            decoded.append(INDEX_TO_CHAR.get(idx, ""))
        prev = idx

    return "".join(decoded)


# ------------------------------
# 🔡 CTC Decoder with Confidence
# ------------------------------
def ctc_decode(preds: torch.Tensor, raw: bool = False) -> List[Tuple[str, float]]:
    """
    Greedy CTC decode with average confidence.
    Args:
        preds (Tensor): shape (T, B, C)
        raw (bool): If True, skip merging repeated characters and removing blanks.
    Returns:
        List of tuples: (decoded_text, average_confidence)
    """
    probs = F.softmax(preds, dim=2)              # (T, B, C)
    max_probs, pred_indices = probs.max(2)       # (T, B)
    pred_indices = pred_indices.permute(1, 0)    # (B, T)
    max_probs = max_probs.permute(1, 0)          # (B, T)

    decoded_batch = []

    for seq, conf_seq in zip(pred_indices, max_probs):
        decoded = []
        confidences = []
        prev_idx = None

        for idx, conf in zip(seq.tolist(), conf_seq.tolist()):
            if raw:
                if 0 < idx <= len(CHARSET):
                    decoded.append(CHARSET[idx - 1])
                    confidences.append(conf)
            else:
                if idx != 0 and idx != prev_idx:
                    if idx <= len(CHARSET):
                        decoded.append(CHARSET[idx - 1])
                        confidences.append(conf)
                prev_idx = idx

        text = "".join(decoded)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        decoded_batch.append((text, avg_conf))

    return decoded_batch


# ------------------------------
# 🧮 Character-Level Accuracy
# ------------------------------
def compute_char_accuracy(preds: list[str], gts: list[str]) -> float:
    correct_chars = 0
    total_chars = 0

    for pred, gt in zip(preds, gts):
        total_chars += len(gt)
        correct_chars += sum(p == g for p, g in zip(pred, gt))

    return correct_chars / total_chars if total_chars > 0 else 0.0