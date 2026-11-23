# MedivUe: U-Net for Medical Image Segmentation

A PyTorch implementation of U-Net for skin lesion segmentation, trained on the ISIC 2018 dataset and evaluated with the Dice coefficient.

## Overview

U-Net is an encoder-decoder convolutional network with skip connections, widely used for biomedical image segmentation. This project trains a U-Net to produce binary lesion masks from dermoscopic images.

![U-Net Architecture](data/images_for_readme/Architecture-of-U-Net-with-dense-block.png)

- **Encoder**: convolutional feature extraction
- **Bottleneck**: bridge between encoder and decoder
- **Decoder**: upsampling and reconstruction of the segmentation mask
- **Skip connections**: retain spatial detail

## Dataset

[ISIC 2018 Challenge, Task 1](https://challenge.isic-archive.com/data/#2018) (lesion boundary segmentation): 16,072 images and masks. Images and masks are resized to 256x256 and converted to tensors. `data/download_isic.py` downloads and extracts the data.

## Project Structure

```
.
├── data/
│   ├── download_isic.py        # Download ISIC 2018 images and masks
│   ├── images_for_readme/      # Figures used in this README
│   └── test/                   # Sample test images
├── src/
│   ├── dataset.py              # PyTorch Dataset class
│   ├── unet.py                 # U-Net model
│   ├── train.py                # Training script (Weights & Biases logging)
│   ├── utils.py                # Dice metric, model save/load
│   └── inference.ipynb         # Inference and visualization
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/m0han-raj/MedivUe.git
cd MedivUe
pip install -r requirements.txt
```

## Usage

Download the dataset:

```bash
cd data
python download_isic.py
```

Train (requires a Weights & Biases login, or set `mode="offline"` in `src/train.py`):

```bash
cd src
python train.py
```

`train.py` currently uses hard-coded dataset and checkpoint paths (`/teamspace/studios/this_studio/...`). Edit `DATASET_PATH`, `MODEL_SAVE_PATH` and the path passed to `save_model` to match your environment.

For evaluation and visualization, open `src/inference.ipynb`.

## Results

Dice coefficient: **88.051%**

**Melanocytic lesion segmentation**

![Melanocytic Lesion](data/images_for_readme/melanocytic_lesion.png)

**Melanoma segmentation**

![Melanoma](data/images_for_readme/melanoma.png)

**Validation Dice score**

![Validation Dice Score](data/images_for_readme/validation_dice_score.png)

**Training loss**

![Training Loss](data/images_for_readme/train_loss.png)

## Tech Stack

Python, PyTorch, torchvision, OpenCV, Matplotlib, Weights & Biases

## References

- Ronneberger et al., [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597), 2015
- [ISIC 2018 Challenge dataset](https://challenge.isic-archive.com/data/#2018)

## License

MIT. See [LICENSE](LICENSE).
