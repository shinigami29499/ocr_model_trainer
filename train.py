import os
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CTCLoss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants.config import *
from model.crnn import CRNN
from ultis.common import *
from ultis.dataset import OCRDataset


# ------------------------------
# 🚂 Training Loop (One Epoch)
# ------------------------------
def train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: Optimizer,
) -> float:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    for images, labels, lengths in tqdm(loader, desc="🔹 Training"):
        images = images.to(DEVICE)
        targets = torch.cat(labels).to(DEVICE)
        target_lengths = torch.tensor(lengths).to(DEVICE)

        outputs = model(images)
        input_lengths = torch.full(
            size=(outputs.size(1),),
            fill_value=outputs.size(0),
            dtype=torch.long,
        ).to(DEVICE)

        loss = criterion(outputs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"🔹 Loss: {avg_loss:.4f}")
    return avg_loss


# ------------------------------
# 🧪 Validation Loop (One Epoch)
# ------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Union[CTCLoss, nn.Module],
    decode_fn,
) -> float:
    """Run one validation epoch. Prints loss and accuracy."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_gts = []

    for images, labels, lengths in tqdm(loader, desc="🔸 Validating"):
        images = images.to(DEVICE)
        targets = torch.cat(labels).to(DEVICE)
        target_lengths = torch.tensor(lengths).to(DEVICE)

        outputs = model(images)
        input_lengths = torch.full(
            size=(outputs.size(1),),
            fill_value=outputs.size(0),
            dtype=torch.long,
        ).to(DEVICE)

        loss = criterion(outputs, targets, input_lengths, target_lengths)
        running_loss += loss.item()

        preds = outputs.permute(1, 0, 2)
        decoded_preds = decode_fn(preds)

        gt_texts = [labels_to_text(label_seq) for label_seq in labels]

        all_preds.extend(decoded_preds)
        all_gts.extend(gt_texts)

    avg_loss = running_loss / len(loader)
    total_chars = sum(len(gt) for gt in all_gts)
    correct_chars = sum(
        sum(p == g for p, g in zip(pred, gt)) for pred, gt in zip(all_preds, all_gts)
    )
    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    word_acc = (
        sum(p == g for p, g in zip(all_preds, all_gts)) / len(all_gts) if all_gts else 0.0
    )

    print(f"🔸 Loss: {avg_loss:.4f}")
    print(f"🎯 Char Accuracy: {char_acc:.2%}")
    print(f"📝 Word Accuracy: {word_acc:.2%}")

    return avg_loss


# ------------------------------
# ♻️ Checkpoint Loader
# ------------------------------
def load_checkpoint(
    model: nn.Module, optimizer: Optimizer, checkpoint_path: str
) -> int:
    """Load model and optimizer state from checkpoint. Returns epoch to resume from."""
    if os.path.exists(checkpoint_path):
        # Dummy forward pass to initialize submodules
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 32, 128).to(DEVICE)
            _ = model(dummy_input)

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"🔁 Resumed from checkpoint: {checkpoint_path} (Epoch {checkpoint['epoch'] + 1})")
        return checkpoint["epoch"] + 1

    print("⚠️ Starting training from scratch")
    return 0


# ------------------------------
# 🚀 Main Training Loop
# ------------------------------
def main() -> None:
    """Main entry point for OCR model training."""
    os.makedirs("checkpoints", exist_ok=True)

    # --- Dataset ---
    train_dataset = OCRDataset(root_dir=TRAIN_DIR)
    val_dataset = OCRDataset(root_dir=VAL_DIR)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # --- Model ---
    model = CRNN(num_classes=len(CHARSET) + 1).to(DEVICE)

    # --- Loss & Optimizer ---
    criterion = CTCLoss(blank=0, zero_infinity=True)
    optimizer = Adam(model.parameters(), lr=LR_SCHEDULE[0])
    lr_index = 0
    optimizer.param_groups[0]['lr'] = LR_SCHEDULE[lr_index]

    best_val_loss = float('inf')
    best_epoch = -1

    # --- Resume from checkpoint ---
    start_epoch = load_checkpoint(model, optimizer, MODEL_PATH) if LOAD_CHECKPOINT else 0

    print()

    patience_counter = 0
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"🧠 Epoch {epoch+1}:")
            print("-" * 40)

            train(model, train_loader, criterion, optimizer)

            # --- Validation ---
            if (epoch + 1) % VAL_CHECK_INTERVAL == 0:
                val_loss = validate(model, val_loader, criterion, decode_fn=ctc_decode)

                # --- Early stopping ---
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"💾 Best model updated")
                else:
                    patience_counter += 1
                    print(f"⏳ No improvement for {patience_counter}/{EARLY_STOPPING_PATIENCE} epochs")

                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        if lr_index < len(LR_SCHEDULE) - 1:
                            lr_index += 1
                            optimizer.param_groups[0]['lr'] = LR_SCHEDULE[lr_index]
                            print(f"🔻 Reducing LR to {optimizer.param_groups[0]['lr']}")
                            patience_counter = 0
                        else:
                            print(f"🛑 Early stopping at LR={optimizer.param_groups[0]['lr']}")
                            break

            # --- Save checkpoint ---
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, MODEL_PATH)

            print()

        print("-" * 40)
        print(f"✅ Training complete. Best model was from epoch {best_epoch+1}.")

    except Exception as e:
        print(f"❌ Training interrupted: {e}")


if __name__ == "__main__":
    main()
