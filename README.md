# IterDiff-PL

IterDiff-PL is a semi-supervised retinal vessel segmentation framework based on diffusion-refined pseudo-labels and uncertainty-weighted distillation. The framework is designed to improve vessel continuity, fine-branch recovery, and pseudo-label reliability when only limited pixel-level annotations are available.

The core idea is to use a segmentation backbone to generate coarse vessel predictions, refine these predictions with a conditional diffusion model, estimate pixel-wise uncertainty through repeated sampling, and distill the refined soft pseudo-labels back into a lightweight segmentation network. During inference, only the student segmentation model is required.

## Highlights

- **Backbone-agnostic segmentation pipeline**  
  The coarse segmentation network can be replaced by different vessel segmentation backbones. The current implementation includes a TransUNet-based model.

- **Diffusion-refined pseudo-labels**  
  Instead of generating masks from pure noise, the diffusion module starts from coarse segmentation results and focuses on local structural repair, especially broken vessels, weak boundaries, and fine branches.

- **Uncertainty-weighted distillation**  
  Multiple diffusion sampling rounds are used to produce soft pseudo-labels and uncertainty maps. Low-confidence regions are down-weighted during student training to reduce pseudo-label noise propagation.

- **Efficient inference**  
  Diffusion refinement is used during training or pseudo-label generation. At deployment time, the distilled student model performs vessel segmentation with a single forward pass.

- **Flexible dataset support**  
  The code can be adapted to standard fundus images, ultra-widefield fundus images, patch-based training, and mixed-resolution training workflows.

## Repository Structure

```text
IterDiff-PL/
├─ model/
│  ├─ transunet.py              # TransUNet-based segmentation backbone
│  ├─ diffusion_model.py        # Conditional diffusion refinement model
│  └─ diffusion_model_v2.py     # Extended diffusion implementation
├─ dataset/
│  ├─ image_process.py          # Patch extraction and preprocessing
│  └─ prepare_diffusion_data.py # Prepare mixed data for diffusion refinement
├─ predict/
│  ├─ generate_transunet_predictions.py # Generate coarse backbone predictions
│  └─ test_diffusion_patch.py           # Patch-based diffusion inference
├─ train_transunet.py           # Supervised backbone training
├─ train_diffusion.py           # Diffusion refiner training and testing menu
├─ semi-supervised.py           # Teacher-Diffusion-Student training framework
├─ requirements.txt
├─ README_CN.md                 # Chinese documentation
└─ README.md
```

## Installation

Create a Python environment and install dependencies:

```bash
conda create -n iterdiff-pl python=3.10
conda activate iterdiff-pl
pip install -r requirements.txt
```

The default dependency file is configured for CUDA 11.8 PyTorch wheels. If your CUDA or PyTorch version is different, install the matching `torch` and `torchvision` versions from the official PyTorch index.

## Runtime Data Layout

The dataset folders, generated pseudo-labels, model checkpoints, and outputs are not included in the repository. They are ignored by `.gitignore` and should be prepared or generated locally when running experiments.

The training scripts expect image and mask folders with paired filenames. A typical local runtime layout is:

```text
IterDiff-PL/
├─ blood-vessel/
│  ├─ train/
│  │  ├─ image/
│  │  └─ label/
│  └─ test/
│     ├─ image/
│     └─ label/
├─ blood-vessel-patch/
│  ├─ train/
│  │  ├─ image/
│  │  └─ label/
│  └─ test/
│     ├─ image/
│     └─ label/
├─ transunet_predictions/
│  ├─ train/
│  └─ test/
└─ weights/
```

Images can be stored as `.png`, `.jpg`, or `.jpeg`. Masks should be binary vessel annotations with the same filenames as the corresponding images.

Prepare these folders locally according to your dataset and checkpoint paths before running the scripts.

## Workflow

### 1. Train the coarse segmentation backbone

```bash
python train_transunet.py
```

This trains the TransUNet-based segmentation model and saves checkpoints under `weights/` or `output/`.

### 2. Generate coarse pseudo-labels

After training or placing a pretrained backbone checkpoint in `weights/`, generate coarse predictions:

```bash
python predict/generate_transunet_predictions.py
```

The generated probability maps are saved as `.npy` files under:

```text
transunet_predictions/train/
transunet_predictions/test/
```

### 3. Prepare diffusion training data

For patch-based or mixed-resolution diffusion refinement data:

```bash
python dataset/prepare_diffusion_data.py
```

This creates a diffusion training set containing images, labels, and coarse predictions.

### 4. Train the diffusion refiner

```bash
python train_diffusion.py
```

The script provides an interactive menu for:

```text
1. Train diffusion model from scratch
2. Resume diffusion training from checkpoint
3. Test diffusion model with patch inference
4. Generate backbone predictions
5. Prepare mixed diffusion training data
```

Diffusion checkpoints are saved under `weights/`.

### 5. Semi-supervised Teacher-Diffusion-Student training

The semi-supervised framework is implemented in:

```bash
python semi-supervised.py
```

The training process follows three stages:

1. Train the student model on labeled data.
2. Use the EMA teacher and diffusion refiner to generate refined soft pseudo-labels for unlabeled data.
3. Distill reliable pseudo-label regions into the student model using uncertainty weighting.

During inference, only the student segmentation model is needed.

## Method Overview

The framework contains three main components:

```text
Input image
   │
   ▼
Coarse segmentation backbone
   │
   ▼
Coarse vessel probability map
   │
   ▼
Conditional diffusion refiner
   │
   ├─ repeated sampling
   ├─ soft pseudo-label estimation
   └─ uncertainty estimation
   │
   ▼
Uncertainty-weighted distillation
   │
   ▼
Student segmentation model
```

The diffusion model is used as a structural pseudo-label refiner rather than a standalone inference model. This design keeps the final model efficient while preserving the benefits of diffusion-based local vessel repair.

## Evaluation Metrics

The code includes common retinal vessel segmentation metrics:

- Dice coefficient
- Intersection over Union, IoU
- Sensitivity / Recall
- Specificity
- Accuracy
- Precision
- AUC-ROC
- AUC-PR

These metrics are used to evaluate coarse segmentation results, diffusion-refined outputs, and final student predictions.

## Customization

### Replace the segmentation backbone

The framework can be adapted to other segmentation backbones by replacing the model used to generate coarse predictions. The expected output is a single-channel vessel probability map.

### Use another dataset

To use a new dataset, organize images and masks using the expected folder structure, or modify the dataset class in the corresponding training script.

### Adjust image size

Default scripts commonly use `512 x 512` images or patches. For high-resolution fundus or ultra-widefield images, patch-based preprocessing is recommended to reduce memory usage.

## Notes

- The diffusion model requires coarse predictions from a trained segmentation backbone.
- For large images, patch-based inference is recommended.
- The final distilled student model is suitable for efficient deployment because it does not require diffusion sampling during inference.
- Some scripts contain fixed default paths. Please check dataset and checkpoint paths before running experiments.

## License

This repository is intended for academic research and experimental use. Please follow the licenses of the datasets and third-party dependencies used in your experiments.
