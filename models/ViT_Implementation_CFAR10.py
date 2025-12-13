import math
import os
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from tqdm import tqdm

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ---------------------------
# Patch embedding module
# ---------------------------
class PatchEmbed(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        # conv projects each patch to embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# ---------------------------
# Positional encodings
# ---------------------------
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for class token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


def get_sincos_pos_embed(embed_dim, grid_size):
    # adapted from common ViT implementations
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w, h
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # grid: 2 x 1 x H x W
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # pos: 1 x H x W
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega = 1. / (10000 ** (omega / (embed_dim / 2)))
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class SinCosPositionalEncoding(nn.Module):
    def __init__(self, grid_size: int, embed_dim: int):
        super().__init__()
        pos_embed = get_sincos_pos_embed(embed_dim, grid_size)
        # pos_embed is shape [num_patches, embed_dim]
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        # add class token
        cls_token = torch.zeros(1, 1, embed_dim)
        self.register_buffer('pos_embed', torch.cat([cls_token, pos_embed], dim=1))

    def forward(self, x):
        # x: [B, num_patches+1, dim]
        return x + self.pos_embed

# ---------------------------
# Transformer Encoder Block with Multi-Head Self-Attention
# ---------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_p, batch_first=True)
        self.drop_path = nn.Identity()  # optionally implement stochastic depth
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # x: [B, N, D]
        x_norm = self.norm1(x)
        # attn: [B, N, N] (averaged across heads by default if average_attn_weights is not set to False)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weights

# ---------------------------
# Vision Transformer
# ---------------------------
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 num_classes: int = 10,
                 embed_dim: int = 128,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 use_sincos: bool = False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # positional encoding
        if use_sincos:
            self.pos_enc = SinCosPositionalEncoding(self.patch_embed.grid_size, embed_dim)
        else:
            self.pos_enc = LearnedPositionalEncoding(num_patches, embed_dim)

        # transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, return_attention: bool = False):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        # add positional encoding
        x = self.pos_enc(x)

        attentions = []
        for blk in self.blocks:
            x, attn = blk(x)
            # attn: [B, N, N] (averaged across heads, as `average_attn_weights=False` was removed)
            attentions.append(attn)

        x = self.norm(x)
        cls = x[:, 0]
        logits = self.head(cls)

        if return_attention:
            return logits, attentions
        return logits

# ---------------------------
# Simple CNN baseline
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_chans=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------------------------
# Metrics: accuracy, ECE
# ---------------------------

def accuracy(output, target):
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()


def compute_ece(probs, labels, n_bins=15):
    # probs: [N, num_classes] (softmaxed)
    # labels: [N]
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)
    return ece.item()

def plot_reliability_diagram(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accs = []
    confs = []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0:
            accs.append(0.0)
            confs.append(bins[i] + (bins[1]-bins[0])/2.0)
        else:
            accs.append(float(accuracies[mask].mean()))
            confs.append(float(confidences[mask].mean()))

    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1], '--')
    plt.plot(confs, accs, marker='o')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True)
    plt.show()


# ---------------------------
# Training and evaluation loops
# ---------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, grad_clip=None):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(dataloader, desc=f'Train Epoch {epoch}', leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device, return_probs=False):
    model.eval()
    total = 0
    correct = 0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Eval', leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
    acc = correct / total
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    ece = compute_ece(all_probs, all_targets)
    if return_probs:
        return acc, ece, all_probs, all_targets
    return acc, ece

# ---------------------------
# Attention visualization utilities
# ---------------------------

def attention_rollout(attentions, discard_ratio=0.0):
    # attentions: list of attention tensors from each block: each is [B, N, N] (averaged across heads)
    result = None
    for attn in attentions:
        # attn: [B, N, N] - already averaged across heads from MultiheadAttention
        # No need for attn.mean(axis=1) here
        attn = attn + torch.eye(attn.size(-1), device=attn.device).unsqueeze(0)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        if result is None:
            result = attn
        else:
            result = torch.matmul(result, attn)
    return result  # [B, N, N]


def visualize_attention_on_image(img: np.ndarray, rollout: np.ndarray, patch_size: int, grid_size: int, save_path: Optional[str]=None):
    # img: H x W x C (0-1)
    # rollout: N x N (for a single sample), we will take attention from class token to patches
    # extract cls->patches
    cls_attn = rollout[0, 1:]  # skip class token self-attn
    # reshape to grid
    attn_map = cls_attn.reshape(grid_size, grid_size)
    # upsample to image size
    attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)
    attn_map = F.interpolate(attn_map, size=(img.shape[0], img.shape[1]), mode='bilinear', align_corners=False)
    attn_map = attn_map.squeeze().numpy()

    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].imshow(img)
    ax[0].set_title('Input image')
    ax[0].axis('off')

    ax[1].imshow(img)
    ax[1].imshow(attn_map, cmap='jet', alpha=0.5)
    ax[1].set_title('Attention overlay')
    ax[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ---------------------------
# Datasets / Dataloaders
# ---------------------------

def get_dataloaders(dataset_name='CIFAR10', batch_size=128, img_size=32):
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        in_chans = 3
        num_classes = 10
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = MNIST(root='./data', train=True, download=True, transform=transform)
        testset = MNIST(root='./data', train=False, download=True, transform=transform)
        in_chans = 1
        num_classes = 10
    else:
        raise ValueError('Unknown dataset')

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader, in_chans, num_classes

# ---------------------------
# Main experiment function
# ---------------------------

def run_experiment(dataset='MNIST',
                   model_type='vit',
                   epochs=10,
                   batch_size=128,
                   lr=3e-4,
                   weight_decay=0.0,
                   img_size=32,
                   patch_size=4,
                   embed_dim=128,
                   depth=12,
                   num_heads=8,
                   use_sincos=False,
                   visualize_attention=True):
    train_loader, test_loader, in_chans, num_classes = get_dataloaders(dataset, batch_size, img_size)

    if model_type == 'vit':
        model = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                  num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                                  num_heads=num_heads, use_sincos=use_sincos).to(device)
    elif model_type == 'cnn':
        model = SimpleCNN(num_classes=num_classes, in_chans=in_chans).to(device)
    elif model_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        if in_chans != 3:
            model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to(device)
    else:
        raise ValueError('Unknown model_type')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_acc, val_ece = evaluate(model, test_loader, device)
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_ece={val_ece:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{dataset}_{model_type}.pth')

    print('Training complete. Best val acc:', best_acc)

    # final evaluation with probabilities
    val_acc, val_ece, probs, targets = evaluate(model, test_loader, device, return_probs=True)
    print('Final: acc=%.4f ece=%.4f' % (val_acc, val_ece))

    # Optionally visualize attention for ViT
    if visualize_attention and model_type == 'vit':
        model.eval()
        # pick a single batch
        imgs, labels = next(iter(test_loader))
        imgs = imgs.to(device)[:8]
        labels = labels[:8]
        logits, attentions = model(imgs, return_attention=True)
        preds = logits.argmax(dim=1) # Define preds
        # attentions: list of [B, N, N]  (averaged across heads)
        rollout = attention_rollout(attentions)  # [B, N, N]
        imgs_cpu = imgs.cpu()

        # Define CIFAR-10 class names, needed for plotting
        if dataset == 'CIFAR10':
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset == 'MNIST': # MNIST has no specific class names like CIFAR-10
            class_names = [str(i) for i in range(10)]
        else:
            class_names = ['Unknown'] * num_classes

        # Plot 10 test images with predictions
        plt.figure(figsize=(12, 4))
        for i in range(min(20, imgs.size(0))): # Ensure we don't try to plot more than available images
          plt.subplot(4, 5, i+1)
          # Permute dimensions from (C, H, W) to (H, W, C) for matplotlib
          # Clip values to [0, 1] to suppress 'Clipping input data' warnings
          img_to_plot = imgs_cpu[i].permute(1, 2, 0).numpy()
          plt.imshow(np.clip(img_to_plot, 0, 1))
          # Ensure preds and labels are correctly indexed
          plt.title(f"Pred: {class_names[preds[i].item()]}\nTrue: {class_names[labels[i].item()]}")
          plt.axis('off')
        plt.tight_layout()
        plt.show()
        # convert and denormalize for plotting (works for CIFAR default normalization)
        for i in range(min(4, imgs_cpu.size(0))):
            img = imgs_cpu[i]
            if img.size(0) == 3:
                # denorm CIFAR
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
                std = torch.tensor([0.247, 0.243, 0.261]).view(3,1,1)
                img = img * std + mean
                npimg = img.permute(1,2,0).numpy()
            else:
                # MNIST
                img = img * 0.3081 + 0.1307
                npimg = img.squeeze(0).numpy()
                npimg = np.stack([npimg]*3, axis=2)
            r = rollout[i].cpu().detach().numpy()
            visualize_attention_on_image(npimg, r, patch_size, img_size // patch_size)

    return model, best_acc, val_ece

# ---------------------------
# If run as a script, run a small default experiment
# ---------------------------
if __name__ == '__main__':
    # quick defaults suitable for Colab CPU/GPU short runs
    dataset = 'CIFAR10'  # 'MNIST' or 'CIFAR10'
    print('Starting experiment on', dataset)
    model, best_acc, val_ece = run_experiment(dataset=dataset,
                                              model_type='vit',
                                              epochs=10,
                                              batch_size=256,
                                              lr=3e-4,
                                              img_size=32,
                                              patch_size=16,
                                              embed_dim=768,
                                              depth=12,
                                              num_heads=12,
                                              use_sincos=False,
                                              visualize_attention=True)

    # compare with CNN baseline
    print('\nRunning CNN baseline...')
    cnn_model, cnn_best_acc, cnn_val_ece = run_experiment(dataset=dataset,
                                                          model_type='cnn',
                                                          epochs=10,
                                                          batch_size=256,
                                                          lr=1e-3,
                                                          img_size=32,
                                                          visualize_attention=False)

    print('Done. ViT best acc:', best_acc, 'ViT val_ece:', val_ece)
    print('CNN best acc:', cnn_best_acc, 'CNN val_ece:', cnn_val_ece)

    print('Models saved as best_<dataset>_<model>.pth')
