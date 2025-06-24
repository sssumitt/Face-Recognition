# Face Recognition Model with ArcFace Loss

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.7-red.svg)](https://pytorch.org/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

A PyTorch-based face recognition framework leveraging ArcFace (Additive Angular Margin) loss to learn highly discriminative embeddings. Train on your own dataset or fine‑tune on popular face benchmarks.

---

## 🔍 Features

* **ArcFace Loss**: Additive angular margin penalty to maximize inter-class separability and intra-class compactness.
* **Modular Design**:

  * `datamodule.py` — data loading, preprocessing, and augmentation.
  * `model.py`      — backbone and embedding head definitions.
  * `losses.py`     — ArcFace and auxiliary loss implementations.
  * `train.py`      — training loop with checkpointing and logging.
  * `inference.py`  — extract embeddings or perform face verification.
* **Configurable**: Custom batch size, learning rate, embedding dimension, margin, and scale via CLI arguments.
* **Dependency Management** via `requirements.txt`.

---

## 🚀 Installation

```bash
git clone https://github.com/sssumitt/Face-Recognition.git
cd Face-Recognition
python3 -m venv venv
source venv/bin/activate      # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

---

## 💡 Usage

### 1. Training

```bash
python train.py \
  --data-dir path/to/face_dataset \
  --batch-size 64 \
  --epochs 50 \
  --lr 0.001 \
  --embedding-size 512 \
  --margin 0.5 \
  --scale 30
```

Key arguments:

* `--data-dir`      : root folder of training data (organized by person)
* `--batch-size`    : number of images per batch
* `--epochs`        : total training epochs
* `--lr`            : initial learning rate
* `--embedding-size`: dimensionality of output embedding
* `--margin`        : angular margin in ArcFace loss
* `--scale`         : feature scale in ArcFace loss

### 2. Inference

To compute an embedding or verify a pair of images:

```bash
# Extract embedding
python inference.py --model-path checkpoints/model.pth --image path/to/image.jpg

# Verify two images
python inference.py --model-path checkpoints/model.pth \
                     --image1 img1.jpg --image2 img2.jpg
```

The script will output cosine similarity score and a decision threshold.

---

## 📂 Project Structure

```plaintext
Face-Recognition/
├── datamodule.py      # Data loading & preprocessing
├── losses.py          # ArcFace (and auxiliary) loss definitions
├── model.py           # Network architecture and embedding head
├── train.py           # Training & checkpointing logic
├── inference.py       # Embedding extraction & verification
├── requirements.txt   # Python dependencies
├── LICENSE            # Apache 2.0 license
└── README.md          # This document
```

---

## 📜 License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
