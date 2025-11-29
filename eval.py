# eval.py
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from models.vit import ViT

# ---------- Data ----------

def get_test_loader(batch_size=256, num_workers=4):
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return testloader

# ---------- Calibration metrics ----------

def compute_ece(probs, labels, n_bins=15):
    """
    probs: (N, num_classes) tensor of predicted probabilities
    labels: (N,) tensor of true labels
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = accuracies[mask].float().mean()
        bin_prob = mask.float().mean()
        ece += torch.abs(bin_confidence - bin_accuracy) * bin_prob

    return ece.item()

def reliability_diagram(probs, labels, n_bins=15, title="Reliability Diagram", out_path=None):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    xs = []
    ys = []
    for i in range(n_bins):
        mask = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        if mask.sum() == 0:
            continue
        xs.append(confidences[mask].mean().item())
        ys.append(accuracies[mask].float().mean().item())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.scatter(xs, ys, label="Bins")
    plt.bar(xs, np.array(ys) - np.array(xs), width=0.03, alpha=0.3)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()

# ---------- Evaluation ----------

def evaluate_model(checkpoint_path, patch_size=4, embed_dim=256, depth=6,
                   heads=4, mlp_dim=512, batch_size=256, n_bins=15):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    testloader = get_test_loader(batch_size=batch_size)

    model = ViT(
        img_size=32,
        patch_size=patch_size,
        in_chans=3,
        num_classes=10,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=heads,
        mlp_dim=mlp_dim,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_probs = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs)
            all_labels.append(labels)

            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy = correct / total
    ece = compute_ece(all_probs, all_labels, n_bins=n_bins)
    reliability_diagram(
        all_probs.cpu(), all_labels.cpu(), n_bins=n_bins,
        title=f"ViT CIFAR-10 (Acc={accuracy:.3f}, ECE={ece:.3f})",
        out_path="reliability_vit_cifar10.png",
    )

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best_vit_cifar10.pth")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.ckpt,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
    )
