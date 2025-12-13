# Vision Transformer (ViT) and Simple CNN for Image Classification

This project implements a Vision Transformer (ViT) and a Simple Convolutional Neural Network (CNN) for image classification tasks, primarily demonstrated on CIFAR-10 and MNIST datasets. It provides a flexible framework to train, evaluate, and visualize the performance of these models, including attention mechanisms for ViT.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Features
- **Vision Transformer (ViT)**: Implementation of a basic Vision Transformer architecture for image classification.
- **Simple CNN Baseline**: A straightforward CNN model for comparison.
- **Positional Encodings**: Supports both Learned Positional Encodings and Sinusoidal Positional Encodings (SinCos).
- **Attention Visualization**: For ViT models, it includes utilities to visualize attention maps overlaid on input images.
- **Dataset Handling**: Configured for CIFAR-10 and MNIST datasets with standard data augmentations and normalization.
- **Training & Evaluation Loops**: Standard PyTorch training and evaluation procedures.
- **Metrics**: Calculates accuracy and Expected Calibration Error (ECE) to assess model performance and calibration.
- **GPU Acceleration**: Utilizes CUDA if available, otherwise falls back to CPU.

## Setup

This project is designed to run in a Google Colab environment, but can also be adapted for local execution with a suitable Python environment.

### Dependencies
The following Python libraries are required:
- `numpy`
- `matplotlib`
- `torch`
- `torchvision`
- `tqdm`

To install these dependencies in your Colab notebook or local environment, run:
```bash
pip install numpy matplotlib torch torchvision tqdm
```

## Usage

The main functionality is encapsulated within the `run_experiment` function, which allows you to configure and execute training for either a ViT or CNN model.

### Running an Experiment

The `if __name__ == '__main__':` block at the end of the script demonstrates how to run experiments with default parameters. You can modify these parameters to customize your experiments.

```python
if __name__ == '__main__':
    # Example default parameters (modify these as needed)
    dataset = 'CIFAR10'  # 'MNIST' or 'CIFAR10'
    batch_size = 256
    img_size = 32
    epochs = 6
    lr = 3e-4

    # ViT specific parameters
    patch_size = 4
    embed_dim = 128
    depth = 6
    num_heads = 8
    use_sincos = False
    visualize_attention = True # Set to False for CNN or if attention visualization is not desired

    # Get dataloaders for the chosen dataset
    train_loader, test_loader, in_chans, num_classes = get_dataloaders(dataset, batch_size, img_size)

    print('Starting experiment on', dataset)
    # Run ViT experiment
    vit_model, vit_best_acc, vit_val_ece = run_experiment(train_loader, test_loader, in_chans, num_classes,
                                                          dataset=dataset,
                                                          model_type='vit', # 'vit', 'cnn', or 'resnet18'
                                                          epochs=epochs,
                                                          batch_size=batch_size,
                                                          lr=lr,
                                                          img_size=img_size,
                                                          patch_size=patch_size,
                                                          embed_dim=embed_dim,
                                                          depth=depth,
                                                          num_heads=num_heads,
                                                          use_sincos=use_sincos,
                                                          visualize_attention=visualize_attention)

    # Run CNN baseline experiment
    print('\nRunning CNN baseline...')
    cnn_model, cnn_best_acc, cnn_val_ece = run_experiment(train_loader, test_loader, in_chans, num_classes,
                                                          dataset=dataset,
                                                          model_type='cnn',
                                                          epochs=epochs,
                                                          batch_size=batch_size,
                                                          lr=1e-3, # CNN might need a different learning rate
                                                          img_size=img_size,
                                                          visualize_attention=False)

    print('Done. ViT best acc:', vit_best_acc, 'ViT val_ece:', vit_val_ece)
    print('CNN best acc:', cnn_best_acc, 'CNN val_ece:', cnn_val_ece)

    print('Models saved as best_<dataset>_<model>.pth')
```

### `run_experiment` Parameters:
- `dataset`: Name of the dataset ('CIFAR10' or 'MNIST').
- `model_type`: Type of model to train ('vit', 'cnn', or 'resnet18').
- `epochs`: Number of training epochs.
- `batch_size`: Batch size for dataloaders.
- `lr`: Learning rate for the optimizer.
- `weight_decay`: Weight decay for the optimizer (default: 0.0).
- `img_size`: Input image size (e.g., 32 for CIFAR-10).
- `patch_size`: (ViT only) Size of patches to divide the image into (e.g., 4, 8, 16).
- `embed_dim`: (ViT only) Dimensionality of the patch embeddings.
- `depth`: (ViT only) Number of transformer encoder blocks.
- `num_heads`: (ViT only) Number of attention heads in the multi-head attention mechanism.
- `use_sincos`: (ViT only) Boolean, `True` to use Sinusoidal Positional Encodings, `False` for Learned Positional Encodings.
- `visualize_attention`: Boolean, `True` to visualize attention maps (only for ViT models).

## Code Structure

The code is organized into several logical sections:

1.  **Utilities**: Contains helper functions like `set_seed`.
2.  **Patch Embedding Module (`PatchEmbed`)**: Transforms input images into sequences of patch embeddings for ViT.
3.  **Positional Encodings**: 
    - `LearnedPositionalEncoding`: Implements learnable positional embeddings.
    - `get_sincos_pos_embed`, `get_2d_sincos_pos_embed_from_grid`, `get_1d_sincos_pos_embed_from_grid`: Functions for generating sinusoidal positional embeddings.
    - `SinCosPositionalEncoding`: Wraps sinusoidal positional embeddings.
4.  **Transformer Encoder Block (`TransformerEncoderBlock`)**: A single block of the ViT encoder, including multi-head self-attention and an MLP.
5.  **Vision Transformer (`VisionTransformer`)**: The main ViT model, combining patch embedding, positional encodings, transformer blocks, and a classification head.
6.  **Simple CNN Baseline (`SimpleCNN`)**: A basic CNN architecture for comparative analysis.
7.  **Metrics**: 
    - `accuracy`: Calculates classification accuracy.
    - `compute_ece`: Computes Expected Calibration Error.
    - `plot_reliability_diagram`: Visualizes model calibration.
8.  **Training and Evaluation Loops**: 
    - `train_one_epoch`: Handles the training process for one epoch.
    - `evaluate`: Evaluates the model on a given dataloader.
9.  **Attention Visualization Utilities**: 
    - `attention_rollout`: Implements the attention rollout mechanism to derive a single attention map.
    - `visualize_attention_on_image`: Overlays attention maps on input images for visual interpretation.
10. **Datasets / Dataloaders**: `get_dataloaders` function to load and preprocess CIFAR-10 or MNIST datasets.
11. **Main Experiment Function (`run_experiment`)**: Orchestrates the entire training, evaluation, and optional visualization process.

## Results

After running an experiment, the script will output training loss, validation accuracy, and Expected Calibration Error (ECE) per epoch. It will also print the best validation accuracy achieved during training and the final validation accuracy and ECE. If `visualize_attention` is `True` for ViT, it will display plots of predictions and attention maps.

Example Output:
```
Using device: cuda
Starting experiment on CIFAR10
Epoch 1: train_loss=1.8566 val_acc=0.3722 val_ece=0.0308
Epoch 2: train_loss=1.5423 val_acc=0.4830 val_ece=0.0085
...
Training complete. Best val acc: 0.5923
Final: acc=0.5923 ece=0.0386

Running CNN baseline...
Epoch 1: train_loss=1.8338 val_acc=0.3960 val_ece=0.0297
Epoch 2: train_loss=1.5316 val_acc=0.4748 val_ece=0.0119
...
Training complete. Best val acc: 0.6171
Final: acc=0.6171 ece=0.0187
Done. ViT best acc: 0.5923 ViT val_ece: 0.038584548979997635
CNN best acc: 0.6171 CNN val_ece: 0.018685081973671913
Models saved as best_CIFAR10_vit.pth
```

## Acknowledgements

- Attention rollout implementation adapted from [eladrich/pytorch-vit-explain](https://github.com/eladrich/pytorch-vit-explain).
