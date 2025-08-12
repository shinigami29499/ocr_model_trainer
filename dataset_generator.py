import concurrent.futures
import os
import random
import shutil
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from constants.config import *


# ------------------------------
# 🔤 Load Fonts from Directory
# ------------------------------
def load_fonts(fonts_dir: str) -> List[ImageFont.FreeTypeFont]:
    """
    Loads all TTF/OTF fonts from a directory into PIL font objects.
    """
    fonts = []
    for file in os.listdir(fonts_dir):
        if file.lower().endswith((".ttf", ".otf")):
            try:
                font_path = os.path.join(fonts_dir, file)
                fonts.append(ImageFont.truetype(font_path, 22))  # Fixed font size
            except Exception as e:
                print(f"❌ Failed to load font {file}: {e}")
    return fonts


# ------------------------------
# 🖼️ Render Text to a Centered Image
# ------------------------------
def render_text_image(
    text: str,
    font: ImageFont.FreeTypeFont
) -> Image.Image:
    """
    Render text in the exact center of a fixed-size image without cropping.
    """
    # Step 1: Create the image
    img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    # Step 2: Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Step 3: Calculate centered position
    x = (TARGET_WIDTH - text_w) // 2
    y = (TARGET_HEIGHT - text_h) // 2

    # Step 4: Draw text
    draw.text((x, y), text, font=font, fill="black")

    return img


def generate_dataset(
    output_dir: str,
    is_predict_data: bool = False,
    num_samples: int = None  # New parameter for predict data sample count
) -> None:
    """
    Generates a synthetic OCR dataset of rendered text images.
    Clears previous contents before generation.
    For predict data, generates only `num_samples` random samples from SEED_WORD.
    """
    images_dir = os.path.join(output_dir, "images")

    # Clear old dataset
    if os.path.exists(output_dir):
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        label_path = os.path.join(output_dir, "labels.txt")
        if os.path.exists(label_path):
            os.remove(label_path)

    os.makedirs(images_dir, exist_ok=True)

    # Load fonts
    fonts = load_fonts(FONTS_DIR)
    if not fonts:
        raise RuntimeError("No fonts found in FONTS_DIR.")

    labels = []

    if is_predict_data and num_samples is not None:
        # Pick num_samples random seed words, use the first font only (same as before)
        chosen_texts = random.choices(SEED_WORD, k=num_samples)
        data = [{"text": text, "font": fonts[0]} for text in chosen_texts]
    else:
        # Original behavior: generate all SEED_WORD * fonts combinations
        data = [
            {"text": char, "font": (fonts[0] if is_predict_data else font)}
            for char in SEED_WORD
            for font in ([None] if is_predict_data else fonts)
        ]

    # Generate images
    for i, sample in enumerate(tqdm(data, desc=f"📦 Generating {output_dir}")):
        image = render_text_image(sample["text"], sample["font"])
        # Augment image: random rotation, blur, noise
        import random

        from PIL import ImageFilter
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, expand=1, fillcolor=255)
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
        if random.random() < 0.3:
            import numpy as np
            arr = np.array(image)
            noise = np.random.normal(0, 10, arr.shape).astype(np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)
        filename = f"{i:06}.png"
        image.save(os.path.join(images_dir, filename))
        labels.append(f"{filename}\t{sample['text']}")

    # Save labels
    with open(os.path.join(output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    print(f"✅ Generated {len(data)} samples in {output_dir}")



# ------------------------------
# 🚦 Entry Point
# ------------------------------
if __name__ == "__main__":
    # Synchronous generation
    generate_dataset(TRAIN_DIR)
    generate_dataset(VAL_DIR, True)
    generate_dataset(PREDICT_DIR, True, 5)
