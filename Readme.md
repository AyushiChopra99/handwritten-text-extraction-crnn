# Handwritten Text Extraction using CRNN + CTC

An image processing project that extracts handwritten text from images using a
Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal
Classification (CTC) loss on the IAM Handwriting Word Database.

## Overview

This project implements an end-to-end pipeline that takes an image containing
handwritten words and outputs the recognized text, without requiring manual
character-level segmentation. It combines:

- **CNN layers** for visual feature extraction from word images
- **Bidirectional LSTM layers** for sequence modeling across the image width
- **CTC loss** to align variable-length text predictions to the image without
  needing per-character bounding boxes

## Dataset

- **Source:** [IAM Handwriting Word Database](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database) (Kaggle mirror)
- **Samples used:** ~38,300 word images after filtering invalid/corrupted files
- **Vocabulary:** 77 characters (uppercase, lowercase, digits, punctuation)

## Model Architecture

```
Input (128 x 32 grayscale image)
  → Conv2D(32) → MaxPool(height only)
  → Conv2D(64) → MaxPool(height only)
  → Conv2D(128) → BatchNorm → MaxPool(height only)
  → Conv2D(128) → BatchNorm
  → Reshape (preserves 128 time steps for CTC)
  → Dense(64) → Dropout
  → Bidirectional LSTM(128)
  → Bidirectional LSTM(64)
  → Dense(vocab_size + 1, softmax)  [+1 for CTC blank token]
```

**Total parameters:** ~646K

## Results

| Metric | Value |
|---|---|
| Epochs trained | 5 |
| Final training loss | 11.98 |
| Final validation loss | 12.89 |
| Character Error Rate (CER) | 0.617 |
| Word-level accuracy | 23.2% (889/3831) |

> **Note:** Due to compute/time constraints, the model was trained for only 5
> epochs. CRNN+CTC models typically require 30–50+ epochs to converge well on
> this dataset. The loss curve (training vs. validation, see
> `results/loss_curve.png`) shows the model was still actively learning and
> had not plateaued when training stopped. This project prioritizes a working,
> reproducible end-to-end pipeline and honest error analysis over a polished
> final accuracy figure.

## Error Analysis

The model performs reasonably on short, clearly-spaced printed-style words
from the training distribution, but accuracy drops noticeably on:
- Cursive or joined handwriting
- Longer words
- Images with uneven lighting/contrast compared to the training data (e.g.
  photos of real handwritten notes vs. scanned IAM samples)

See `results/sample_predictions.png` for qualitative examples, including
both correct and incorrect predictions.

## Project Structure

```
├── notebooks/
│   └── handwriting_crnn.ipynb      # Full Colab notebook (data → train → eval)
├── models/
│   └── crnn_best.keras             # Trained Keras model checkpoint
├── results/
│   ├── loss_curve.png
│   └── sample_predictions.png
├── requirements.txt
└── README.md
```

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/AyushiChopra99/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `notebooks/handwriting_crnn.ipynb` in Google Colab or Jupyter and run
cells top to bottom. The notebook covers:
1. Dataset download (via `kagglehub`) and preprocessing
2. Vocabulary building and label encoding
3. `tf.data` pipeline construction
4. CRNN model definition and training
5. Evaluation (CER / word accuracy)
6. Inference on custom uploaded images

### 4. Run inference on your own image
```python
import tensorflow as tf
from utils import preprocess_image, ctc_decode_predictions  # if extracted to a script

model = tf.keras.models.load_model("models/crnn_best.keras", compile=False)
# ... see notebook Cell 17 for full inference code
```

## Tech Stack

- Python, TensorFlow / Keras
- OpenCV, PIL (image preprocessing)
- Google Colab (training environment, GPU)

## Future Improvements

- Train for significantly more epochs (30–50+) to reach target accuracy
- Use beam search decoding instead of greedy CTC decoding for better accuracy
- Expand training data with augmentation (rotation, noise, varying stroke width)
- Export the model to TensorFlow Lite and build Android integration for on-device inference

## Author

**Ayushi Chopra**
BCA (Big Data Analytics), Parul University
[LinkedIn](https://linkedin.com/in/ayushi-chopra) · [GitHub](https://github.com/AyushiChopra99)

## License

This project is for academic purposes. The IAM Handwriting Database is used
under its own license terms — see [IAM Database](https://fki.tuwien.ac.at/databases/iam-handwriting-database/)
for details.
