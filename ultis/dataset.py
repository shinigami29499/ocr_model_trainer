import os
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset

from ultis.common import preprocess_image  # make sure this is correctly imported


# ----------------------------------------
# 📁 OCRDataset: Loads image-text OCR data
# ----------------------------------------
class OCRDataset(Dataset):
    """
    Custom Dataset for OCR training. Expects the following directory structure:

        root_dir/
            ├── images/
            │     ├── image1.png
            │     ├── image2.jpg
            └── labels.txt

    The `labels.txt` file must contain:
        image_filename<tab>label_text

    Example:
        hello.png   Hello
        img1.jpg    Welcome123

    Args:
        root_dir (str): Directory containing 'images/' and 'labels.txt'
    """

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): Directory containing 'images/' and 'labels.txt'
        Raises:
            FileNotFoundError: If labels.txt is missing
        """
        self.root_dir = root_dir
        label_path = os.path.join(root_dir, "labels.txt")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"labels.txt not found in {root_dir}")
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        self.samples = []
        for line in lines:
            if not line.strip():
                continue  # Skip empty lines
            if '\t' not in line:
                print(f"Warning: Malformed label line skipped: {line}")
                continue
            img_name, text = line.strip().split("\t", maxsplit=1)
            self.samples.append((img_name, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            image_tensor (Tensor): (1, H, W) preprocessed image
            label (str): Raw label string
        """
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, "images", img_name)

        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess_image(image)

        return image_tensor, label
