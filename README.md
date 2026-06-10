# Food-101 Transfer Learning & Fine-Tuning 🍕

Comparing transfer learning and fine-tuning strategies for image classification on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) using pretrained CNN architectures. Built as part of the Deep Learning subject at UTS Sydney.

## Results

### Part A: Transfer Learning (Frozen Backbone)

Three pretrained architectures were evaluated with only a custom classification head trained:

| Model | Test Accuracy | Test Loss | Training Time | Trainable Params |
|---|---|---|---|---|
| **MobileNet V3** | **0.3593** | 2.59 | 332.6s | 682,085 |
| ResNet18 | 0.2330 | 3.15 | 297.2s | 419,941 |
| GoogLeNet | 0.1715 | 3.53 | 324.2s | 682,085 |

MobileNet V3 achieved the best accuracy despite having a lightweight architecture, demonstrating strong feature transfer from ImageNet.

### Part B: Fine-Tuning (MobileNet V3)

MobileNet V3 was selected for fine-tuning experiments based on Part A results:

| Experiment | Test Accuracy | Test Loss | Val Accuracy |
|---|---|---|---|
| Baseline (Frozen Head) | 0.3593 | 2.59 | 0.3423 |
| Unfreeze Last Block | 0.1196 | 4.09 | 0.1081 |
| Unfreeze Last Two Blocks | 0.1873 | 3.61 | 0.1741 |

Fine-tuning degraded performance under the 2-epoch setup, confirming that unfreezing layers requires longer training and lower learning rates to be effective.

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/): 101 food categories, 101,000 images total (750 train + 250 test per class). Challenging due to high intra-class variation in lighting, presentation, and background.

## Notebooks

| Notebook | Description |
|---|---|
| `Part_A_Transfer_Learning.ipynb` | Frozen backbone comparison: ResNet18, MobileNet V3, GoogLeNet |
| `Part_B_Fine_Tuning.ipynb` | Fine-tuning MobileNet V3 with 1 and 2 unfrozen convolutional blocks |

## Architecture

Each model uses a custom 3-layer classification head replacing the original:

```
Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear (101 classes)
```

**Training config:** Adam optimizer, CrossEntropyLoss, batch size 32, 2 epochs, CUDA GPU.

## Setup

```bash
git clone https://github.com/Mitesh2412/food101-transfer-learning.git
cd food101-transfer-learning

pip install -r requirements.txt
jupyter lab
```

Dataset downloads automatically via `torchvision.datasets.Food101`.

## Key Findings

- MobileNet V3 outperforms ResNet18 and GoogLeNet under frozen backbone setup
- Fine-tuning with only 2 epochs and a standard learning rate hurts performance
- Effective fine-tuning requires longer training, lower learning rates for unfrozen layers, and more data
- Transfer learning from ImageNet is highly effective even with minimal training epochs

---

*Built with PyTorch, torchvision*
