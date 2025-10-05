import os
import math
import time
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T  # only for ToTensor / Normalize if desired
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from torchsummary import summary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 50           # Recommended: 150-300 to reach target accuracy
PRINT_FREQ = 100
SAVE_DIR = "./checkpoints_allconv"
os.makedirs(SAVE_DIR, exist_ok=True)
MIXUP_ALPHA = 0.0      # 0 -> no mixup. Try 0.2 for improved results.
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LR = 0.1
USE_FP16 = True  

# CIFAR-10 pixel means (in 0..1) and stds (for normalization)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

# For CoarseDropout fill_value: albumentations works with 0-255 or 0-1 floats.
# We'll use 0..1 floats (mean).
COARSE_FILL = tuple(CIFAR10_MEAN)

# ---------- Albumentations transforms ----------
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=15, p=0.7, border_mode=0),
    A.CoarseDropout(max_holes=1, max_height=8, max_width=8,
                    min_holes=1, min_height=8, min_width=8,
                    fill_value=COARSE_FILL, mask_fill_value=None, p=0.5),
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD, max_pixel_value=1.0),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD, max_pixel_value=1.0),
    ToTensorV2(),
])


class AlbumentationsCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.alb_transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # img is numpy array HxWxC with uint8 0..255. We want 0..1 floats for our chosen albumentations settings.
        img = img.astype(np.float32) / 255.0
        if self.alb_transform:
            aug = self.alb_transform(image=img)
            img = aug["image"]
        return img, target


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        # depthwise
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride,
                            padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        # pointwise
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        layers = []
        in_ch = 3
        # We'll build 21 conv-like layers. Most are 3x3 with 32 channels.
        # layer idx used for placing the required special layers:
        # - depthwise-separable at idx 2
        # - dilated conv at idx 11 (dilation=2)
        # - final conv at idx 20 with stride=2
        n_layers = 21
        for i in range(n_layers):
            if i == 2:
                # depthwise-separable: in_ch -> 32
                layers.append(DepthwiseSeparable(in_ch, 32, kernel=3, stride=1))
                in_ch = 32
            elif i == 11:
                # dilated conv (dilation=2)
                layers.append(ConvBlock(in_ch, 32, kernel=3, stride=1, dilation=2))
                in_ch = 32
            elif i == (n_layers - 1):
                # final conv uses stride=2 (no pooling). Raise channels a bit.
                layers.append(ConvBlock(in_ch, 64, kernel=3, stride=2, dilation=1))
                in_ch = 64
            else:
                layers.append(ConvBlock(in_ch, 32, kernel=3, stride=1, dilation=1))
                in_ch = 32
        self.features = nn.Sequential(*layers)
        # Global Average Pooling -> FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

        # init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)        # (N, C, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def receptive_field_calc(layers_spec):
    # layers_spec: list of dicts with k, s, d
    rf = 1
    total_stride = 1
    for L in layers_spec:
        k = L.get("k", 1)
        s = L.get("s", 1)
        d = L.get("d", 1)
        effective_k = d * (k - 1) + 1
        rf = rf + (effective_k - 1) * total_stride
        total_stride *= s
    return rf, total_stride


def build_layers_spec_for_rf(n_layers=21):
    spec = []
    for i in range(n_layers):
        if i == 2:
            spec.append({"k":3,"s":1,"d":1})  # depthwise has same RF effect as conv
        elif i == 11:
            spec.append({"k":3,"s":1,"d":2})  # dilated
        elif i == n_layers - 1:
            spec.append({"k":3,"s":2,"d":1})  # final stride=2
        else:
            spec.append({"k":3,"s":1,"d":1})
    return spec
    

def mixup_data(x, y, alpha=1.0, device="cuda"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)


def train_one_epoch(model, loader, optimizer, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # mixup?
        if MIXUP_ALPHA > 0.0:
            images, targets_a, targets_b, lam = mixup_data(images, targets, MIXUP_ALPHA, device=device)
        optimizer.zero_grad()
        if USE_FP16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if MIXUP_ALPHA > 0.0:
                    loss = mixup_criterion(outputs, targets_a, targets_b, lam)
                else:
                    loss = F.cross_entropy(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if MIXUP_ALPHA > 0.0:
                loss = mixup_criterion(outputs, targets_a, targets_b, lam)
            else:
                loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        if MIXUP_ALPHA > 0.0:
            _, predicted = outputs.max(1)
            # for accuracy with mixup we count using the mixed label a (approximate)
            # Here we approximate by comparing to targets (not perfect). Accuracy underestimates slightly.
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        else:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        if (i + 1) % PRINT_FREQ == 0:
            print(f"Epoch {epoch} Iter {i+1}/{len(loader)} loss {(running_loss/total):.4f} acc {(100.*correct/total):.2f}%")
    dur = time.time() - start
    return running_loss / total, 100.0 * correct / total, dur


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100.0 * correct / total


def main():
    print("Device:", DEVICE)
    # Datasets
    train_set = AlbumentationsCIFAR10(root="./data", train=True, download=True, transform=train_transforms)
    test_set = AlbumentationsCIFAR10(root="./data", train=False, download=True, transform=test_transforms)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = CIFAR10Net(num_classes=10).to(DEVICE)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    summary(model, input_size=(3, 32, 32))
    
    # print params and receptive field
    n_params = count_params(model)
    print(f"Model params: {n_params:,}")
    spec = build_layers_spec_for_rf(n_layers=21)
    rf, ts = receptive_field_calc(spec)
    print(f"Receptive field: {rf}, final total_stride: {ts}")

    assert n_params < 200_000, f"Params exceed 200k: {n_params}"
    assert rf > 44, f"Receptive field {rf} is not > 44"

    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_FP16 and DEVICE.startswith("cuda")))

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, dur = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch, scaler=scaler)
        val_loss, val_acc = validate(model, test_loader, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS}  train_loss {train_loss:.4f} train_acc {train_acc:.2f}%  val_loss {val_loss:.4f} val_acc {val_acc:.2f}%  time {dur:.1f}s")
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'best_acc': best_acc},
                       os.path.join(SAVE_DIR, "best_checkpoint.pth"))
            print(f"Saved best checkpoint at epoch {epoch} val_acc {val_acc:.2f}%")

    print("Training complete. Best val acc:", best_acc)


if __name__ == "__main__":
    main()
