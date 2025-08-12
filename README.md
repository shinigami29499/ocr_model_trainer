# OCR Model Trainer

A Python-based project for training, evaluating, and using an Optical Character Recognition (OCR) model. This repository contains scripts and utilities for dataset generation, model training, and text prediction from images.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Generation](#dataset-generation)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project focuses on building and training a CRNN (Convolutional Recurrent Neural Network)-based OCR model capable of recognizing text from images. It supports:

- Synthetic dataset generation using various fonts.
- Training with CTC loss.
- Prediction and decoding of text from images.

The main scripts include:

- `dataset_generator.py` – Generate synthetic images with text for training.
- `train.py` – Train the OCR model using generated or custom datasets.
- `predict.py` – Predict text from input images using a trained model.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch (compatible version with your CUDA or CPU)
- Other dependencies listed below

### Setup

1. Clone this repository:

```bash
git clone https://github.com/shinigami29499/ocr_model_trainer.git
cd ocr_model_trainer
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install required Python packages:

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is missing, install manually the essentials:*

```bash
pip install torch torchvision pillow tqdm numpy
```

---

## Usage

### Dataset Generation

Generate synthetic images with random or predefined text for training the OCR model.

```bash
python dataset_generator.py --output_dir ./data/train --num_samples 10000
```

*Options may vary; check `dataset_generator.py` for arguments.*

### Training the Model

Train the OCR model on your dataset.

```bash
python train.py --data_dir ./data/train --epochs 50 --batch_size 32 --lr 0.001
```

*Common parameters:*

- `--data_dir`: Directory with training images and labels.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--lr`: Learning rate.

Checkpoint models are saved during training.

### Making Predictions

Use a trained model to predict text from images.

```bash
python predict.py --model_path ./checkpoints/best_model.pth --image_path ./test_images/sample.png
```

---

## Model Architecture

The OCR model is based on CRNN architecture which combines convolutional layers for feature extraction with recurrent layers (LSTM or GRU) for sequence modeling, optimized with CTC loss for sequence transcription without requiring aligned labels.

---

## Configuration

Adjust constants and hyperparameters in the `constants/` directory and training scripts as needed, such as:

- Character set and mappings
- Image preprocessing parameters
- Model hyperparameters

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## License

Specify your license here. For example:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If you need help with specific parts or want me to generate a `requirements.txt` or example commands, just ask!
