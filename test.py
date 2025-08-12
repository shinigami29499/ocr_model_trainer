import torch
from PIL import Image
from torchvision import transforms as T

# Import your preprocess_image
from ultis.common import preprocess_image

# Load the same image twice
# Load image
img_path1 = r"E:\0. HaND4\Project\ocr_model_trainer\data\predict\images\000000.png"
img_path2 = r"E:\0. HaND4\Project\ocr_model_trainer\data\predict\images\000001.png"
image1   = Image.open(img_path1).convert("RGB")
image2   = Image.open(img_path2).convert("RGB")

# Preprocess both
tensor1 = preprocess_image(image1)
tensor2 = preprocess_image(image2)

# Check shape
print("Shape:", tensor1.shape)

# Check if they are identical
are_equal = torch.equal(tensor1, tensor2)
print("Identical tensors:", are_equal)

# Optional: visualize
to_pil = T.ToPILImage()
to_pil(tensor1).show(title="Processed Image")
