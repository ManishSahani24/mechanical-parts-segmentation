"""Rebuild notebooks/nppe1_semantic_segmentation.ipynb.

The notebook is the original Kaggle submission, preserved verbatim as the
record of what actually produced the leaderboard score. `src/` is the same
pipeline refactored into modules; this file is not derived from it.
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS = [
    (MD, '''# NPPE-1 — Semantic Segmentation of Mechanical Components

Original Kaggle notebook (Deep Learning & Gen AI course, NPPE-1).
Trained on 2x Tesla T4. **Best validation Dice: 0.9695**, public leaderboard
rank **108 / 708**.

Pipeline: EDA -> augmentation -> from-scratch U-Net with ASPP + deep
supervision -> Dice+Focal loss -> 60 epochs with cosine warm restarts ->
8-fold TTA -> RLE submission.
'''),

    (MD, "## 1. Environment"),
    (CODE, r'''import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''),

    (CODE, r'''import subprocess
subprocess.run(["pip", "install", "albumentations", "--quiet"], check=False)

import torch, torchvision, albumentations
print(f"PyTorch    : {torch.__version__}")
print(f"CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"Albumentations: {albumentations.__version__}")
'''),

    (CODE, r'''import os, math, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
'''),

    (MD, "## 2. Paths & hyper-parameters"),
    (CODE, r'''BASE           = Path("/kaggle/input/competitions/nppe-1-dl-gen-ai-course-semantic-segmentation/Dataset")
TRAIN_IMG_DIR  = BASE / "train" / "images"
TRAIN_MASK_DIR = BASE / "train" / "masks"
TEST_IMG_DIR   = BASE / "test"  / "images"
SAMPLE_SUB     = BASE / "sample_submission.csv"
OUTPUT_DIR     = Path("/kaggle/working")

IMG_SIZE     = 384
NUM_CLASSES  = 7        # 0 background + 6 parts
BATCH_SIZE   = 16        # reduce to 4 if OOM on smaller GPU
NUM_EPOCHS   = 60
LR_INITIAL   = 3e-4
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.15
NUM_WORKERS  = 2

CLASS_NAMES = {0:"background",1:"hex_nut",2:"washer",3:"bolt",
               4:"ball_bearing",5:"spring",6:"o_ring"}
CLASS_COLORS= {0:(30,30,30),1:(255,80,80),2:(80,180,255),3:(255,200,0),
               4:(0,220,120),5:(200,80,255),6:(255,140,40)}
'''),

    (MD, "## 3. Run-length encoding\n\nThe competition uses column-major (Fortran) order with 1-based starts."),
    (CODE, r'''def rle_encode(mask):
    """mask: 384x384 boolean array -> RLE string."""
    px = np.asarray(mask, np.uint8).T.flatten()
    px = np.concatenate([[0], px, [0]])
    runs = np.where(px[1:] != px[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)

def rle_decode(rle, size=384):
    """RLE string -> 384x384 uint8 array."""
    m = np.zeros(size * size, np.uint8)
    if not isinstance(rle, str) or not rle.strip():
        return m.reshape(size, size, order="F")
    a = np.asarray(rle.split(), np.int64)
    for s, l in zip(a[0::2], a[1::2]):
        m[s - 1: s - 1 + l] = 1
    return m.reshape(size, size, order="F")

# sanity check
_m = np.zeros((384,384), np.uint8); _m[5:50, 10:60] = 1
assert np.array_equal(rle_decode(rle_encode(_m)), _m)
print("RLE sanity check passed")
'''),

    (MD, "## 4. Exploratory data analysis"),
    (CODE, r'''train_ids = sorted([p.stem for p in TRAIN_IMG_DIR.glob("*.png")])
test_ids  = sorted([p.stem for p in TEST_IMG_DIR.glob("*.png")])
print(f"Train images: {len(train_ids)}  |  Test images: {len(test_ids)}")

# Per-class pixel statistics
pixel_counts = {c: 0 for c in range(NUM_CLASSES)}
total_pixels = 0
for stem in train_ids[:200]:   # sample 200 for speed
    mask = np.array(Image.open(TRAIN_MASK_DIR / f"{stem}.png"))
    for c in range(NUM_CLASSES):
        pixel_counts[c] += (mask == c).sum()
    total_pixels += mask.size

print("\nClass pixel distribution (sampled from 200 images):")
for c, name in CLASS_NAMES.items():
    pct = 100 * pixel_counts[c] / total_pixels
    print(f"  {c}: {name:<14}  {pct:5.1f}%  ({pixel_counts[c]:,} px)")

# Visualise a few training samples
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
sample_ids = random.sample(train_ids, 3)

def colorise_mask(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    for c, col in CLASS_COLORS.items():
        rgb[mask == c] = col
    return rgb

for row, sid in enumerate(sample_ids):
    img  = np.array(Image.open(TRAIN_IMG_DIR / f"{sid}.png").convert("RGB"))
    mask = np.array(Image.open(TRAIN_MASK_DIR / f"{sid}.png"))
    cmask= colorise_mask(mask)
    overlay = (0.55*img + 0.45*cmask).astype(np.uint8)

    axes[row, 0].imshow(img);     axes[row, 0].set_title(f"{sid} - image")
    axes[row, 1].imshow(cmask);   axes[row, 1].set_title("mask (colourised)")
    axes[row, 2].imshow(overlay); axes[row, 2].set_title("overlay")
    axes[row, 3].axis("off")
    # class legend
    patches = [mpatches.Patch(color=[c/255 for c in col],
                               label=CLASS_NAMES[i])
               for i, col in CLASS_COLORS.items()]
    axes[row, 3].legend(handles=patches, loc="center", fontsize=9, frameon=False)

for ax in axes.flat: ax.axis("off")
plt.suptitle("Training Samples with Ground-Truth Masks", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_samples.png", dpi=100, bbox_inches="tight")
plt.show()
'''),

    (MD, "## 5. Train / validation split"),
    (CODE, r'''rng      = np.random.RandomState(SEED)
val_n    = int(len(train_ids) * VAL_FRAC)
idx      = rng.permutation(len(train_ids))
TRAIN_IDS = [train_ids[i] for i in idx[val_n:]]
VAL_IDS   = [train_ids[i] for i in idx[:val_n]]
print(f"Train: {len(TRAIN_IDS)}  |  Val: {len(VAL_IDS)}")
'''),

    (MD, "## 6. Augmentation"),
    (CODE, r'''MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                       rotate_limit=45, border_mode=0, p=0.6),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),
    A.GridDistortion(p=0.2),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.CLAHE(clip_limit=4.0),
    ], p=0.7),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                    fill_value=0, mask_fill_value=0, p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

val_transform = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
test_transform= A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
'''),

    (MD, "## 7. Dataset & dataloaders"),
    (CODE, r'''class SegDataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir=None, transform=None):
        self.img_ids = img_ids; self.img_dir = img_dir
        self.mask_dir = mask_dir; self.transform = transform

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        stem = self.img_ids[idx]
        img  = np.array(Image.open(self.img_dir / f"{stem}.png").convert("RGB"))
        if self.mask_dir is not None:
            mask = np.array(Image.open(self.mask_dir / f"{stem}.png"))  # <- DO NOT .convert
            if self.transform:
                out = self.transform(image=img, mask=mask)
                return out["image"], out["mask"].long(), stem
            return torch.tensor(img).permute(2,0,1).float()/255., torch.tensor(mask).long(), stem
        else:
            if self.transform:
                out = self.transform(image=img)
                return out["image"], stem
            return torch.tensor(img).permute(2,0,1).float()/255., stem

train_ds = SegDataset(TRAIN_IDS, TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform)
val_ds   = SegDataset(VAL_IDS,   TRAIN_IMG_DIR, TRAIN_MASK_DIR, val_transform)
test_ds  = SegDataset(test_ids,  TEST_IMG_DIR,  mask_dir=None,  transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Batches - train: {len(train_loader)}  val: {len(val_loader)}  test: {len(test_loader)}")
'''),

    (MD, "## 8. Model — from-scratch U-Net with ASPP and deep supervision"),
    (CODE, r'''class ConvBnAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dilation=1, groups=1):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.c1 = ConvBnAct(ch, ch, dilation=dilation)
        self.c2 = ConvBnAct(ch, ch, dilation=dilation)
    def forward(self, x): return x + self.c2(self.c1(x))

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.down = ConvBnAct(in_ch, out_ch, stride=2)
        self.res  = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])
    def forward(self, x): return self.res(self.down(x))

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBnAct(in_ch, out_ch, kernel=1 if r==1 else 3, dilation=r)
            for r in rates])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01), nn.GELU())
        self.proj = ConvBnAct(out_ch*(len(rates)+1), out_ch, kernel=1)
        self.drop = nn.Dropout2d(0.1)
    def forward(self, x):
        feats = [c(x) for c in self.convs]
        feats.append(self.gap(x).expand_as(feats[0]))
        return self.drop(self.proj(torch.cat(feats, dim=1)))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnAct(in_ch + skip_ch, out_ch),
            ConvBnAct(out_ch, out_ch))
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class ScratchUNet(nn.Module):
    """Full U-Net trained from scratch - Kaiming initialisation."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(ConvBnAct(3,64), ConvBnAct(64,64))
        self.enc1 = EncoderBlock(64,  128, n_res=2)
        self.enc2 = EncoderBlock(128, 256, n_res=2)
        self.enc3 = EncoderBlock(256, 512, n_res=3)
        self.enc4 = EncoderBlock(512,1024, n_res=3)
        self.aspp = ASPP(1024, 512, rates=(1,4,8,16))
        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128,  64)
        self.dec1 = DecoderBlock( 64,  64,  64)
        self.head_main = nn.Conv2d(64,  num_classes, 1)
        self.head_aux3 = nn.Conv2d(256, num_classes, 1)
        self.head_aux2 = nn.Conv2d(128, num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0); s2 = self.enc2(s1)
        s3 = self.enc3(s2); s4 = self.enc4(s3)
        b  = self.aspp(s4)
        d4 = self.dec4(b, s3); d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1); d1 = self.dec1(d2, s0)
        out = self.head_main(d1)
        if self.training:
            aux3 = F.interpolate(self.head_aux3(d4), size=(IMG_SIZE,IMG_SIZE),
                                 mode="bilinear", align_corners=False)
            aux2 = F.interpolate(self.head_aux2(d3), size=(IMG_SIZE,IMG_SIZE),
                                 mode="bilinear", align_corners=False)
            return out, aux3, aux2
        return out

model = ScratchUNet().to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

n_p = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_p/1e6:.2f}M")
'''),

    (MD, "## 9. Loss and metric"),
    (CODE, r'''class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        C     = logits.size(1)
        probs = torch.softmax(logits, dim=1)
        tgt   = F.one_hot(targets, C).permute(0,3,1,2).float()
        dims  = (0,2,3)
        inter = (probs[:,1:] * tgt[:,1:]).sum(dims)
        card  = probs[:,1:].sum(dims) + tgt[:,1:].sum(dims)
        return 1.0 - ((2*inter + self.smooth) / (card + self.smooth)).mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, reduction="none")
        pt  = torch.exp(-ce)
        return (((1-pt)**self.gamma)*ce).mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice  = DiceLoss()
        self.focal = FocalLoss(gamma=2.0)
    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.focal(logits, targets)

criterion = CombinedLoss()
'''),

    (CODE, r'''def dice_metric(pred, gt, num_classes=NUM_CLASSES):
    scores = []
    for c in range(1, num_classes):
        p = pred == c; t = gt == c
        d = (p & t).sum(); n = p.sum() + t.sum()
        scores.append(1.0 if n == 0 else 2*d/n)
    return np.mean(scores)
'''),

    (MD, "## 10. Training"),
    (CODE, r'''optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INITIAL, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=LR_MIN)
scaler    = torch.cuda.amp.GradScaler()

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train(); total = 0.0
    for imgs, masks, _ in loader:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out, aux3, aux2 = model(imgs)
            loss = (criterion(out, masks)
                  + 0.4*criterion(aux3, masks)
                  + 0.2*criterion(aux2, masks))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval(); t_loss = 0.0; t_dice = 0.0
    for imgs, masks, _ in loader:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            out  = model(imgs)
            loss = criterion(out, masks)
        preds = out.argmax(1).cpu().numpy()
        gts   = masks.cpu().numpy()
        for p, g in zip(preds, gts): t_dice += dice_metric(p, g)
        t_loss += loss.item() * imgs.size(0)
    n = len(loader.dataset)
    return t_loss/n, t_dice/n
'''),

    (CODE, r'''best_dice = 0.0
ckpt_path = OUTPUT_DIR / "best_model.pth"
history   = {"train_loss":[], "val_loss":[], "val_dice":[]}

for epoch in range(1, NUM_EPOCHS+1):
    tl = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
    vl, vd = validate(model, val_loader, criterion)
    scheduler.step()
    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["val_dice"].append(vd)
    flag = ""
    if vd > best_dice:
        best_dice = vd
        torch.save({"epoch":epoch,"state_dict":model.state_dict(),"val_dice":vd}, ckpt_path)
        flag = "  <- saved"
    lr = optimizer.param_groups[0]["lr"]
    print(f"Ep {epoch:03d}/{NUM_EPOCHS}  Loss {tl:.4f}/{vl:.4f}  Dice {vd:.4f}  LR {lr:.2e}{flag}")

print(f"\nBest val Dice: {best_dice:.4f}")
'''),

    (CODE, r'''fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
ax1.set_title("Combined Loss"); ax1.legend(); ax1.grid(True)
ax2.plot(history["val_dice"], color="green"); ax2.set_title("Validation Dice"); ax2.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"training_curves.png", dpi=120); plt.show()
'''),

    (MD, "## 11. Test-time augmentation"),
    (CODE, r'''@torch.no_grad()
def predict_tta(model, img_tensor):
    """8-fold TTA - average softmax probabilities."""
    model.eval()
    x = img_tensor.unsqueeze(0).to(DEVICE)
    augs  = [x, x.flip(3), x.flip(2), x.flip([2,3]),
             x.rot90(1,[2,3]), x.rot90(2,[2,3]),
             x.rot90(3,[2,3]), x.rot90(1,[2,3]).flip(3)]
    probs = None
    for i, aug in enumerate(augs):
        with torch.cuda.amp.autocast():
            p = torch.softmax(model(aug), 1)
        # inverse transforms
        if i == 1: p = p.flip(3)
        elif i == 2: p = p.flip(2)
        elif i == 3: p = p.flip([2,3])
        elif i == 4: p = p.rot90(3,[2,3])
        elif i == 5: p = p.rot90(2,[2,3])
        elif i == 6: p = p.rot90(1,[2,3])
        elif i == 7: p = p.flip(3).rot90(3,[2,3])
        probs = p if probs is None else probs + p
    return (probs / len(augs)).squeeze(0).cpu()
'''),

    (MD, "## 12. Submission"),
    (CODE, r'''CLASS_LABELS = ["hex_nut","washer","bolt","ball_bearing","spring","o_ring"]
CLS_ID = {n:i+1 for i,n in enumerate(CLASS_LABELS)}

# Load best checkpoint
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded checkpoint - val Dice: {ckpt['val_dice']:.4f}")

sample  = pd.read_csv(SAMPLE_SUB)
results = {}

for batch in test_loader:
    imgs_t, stems = batch
    for i, stem in enumerate(stems):
        probs     = predict_tta(model, imgs_t[i])
        pred_mask = probs.argmax(0).numpy().astype(np.uint8)
        for cls_name in CLASS_LABELS:
            bin_mask = (pred_mask == CLS_ID[cls_name])
            rle      = rle_encode(bin_mask) if bin_mask.any() else "1 1"
            results[f"{stem}.png_{cls_name}"] = rle

sample["EncodedPixels"] = sample["ImageId_ClassId"].map(results).fillna("1 1")
sample.to_csv(OUTPUT_DIR/"submission.csv", index=False)
print(f"submission.csv written  ({len(sample)} rows)")
print(sample.head(12))
'''),

    (MD, "## 13. Qualitative check on the validation split"),
    (CODE, r'''model.eval()
fig, axes = plt.subplots(4, 3, figsize=(12, 15))
shown = 0
for imgs, masks, stems in val_loader:
    for i in range(len(stems)):
        if shown >= 4: break
        img_np  = imgs[i].permute(1,2,0).numpy()
        img_np  = (img_np * np.array(STD) + np.array(MEAN)).clip(0,1)
        gt_mask = masks[i].numpy()
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred_mask = model(imgs[i].unsqueeze(0).to(DEVICE)).argmax(1).squeeze().cpu().numpy()
        gt_col   = colorise_mask(gt_mask)
        pred_col = colorise_mask(pred_mask)
        dice = dice_metric(pred_mask, gt_mask)
        axes[shown,0].imshow(img_np);    axes[shown,0].set_title(f"{stems[i]}")
        axes[shown,1].imshow(gt_col);    axes[shown,1].set_title("Ground Truth")
        axes[shown,2].imshow(pred_col);  axes[shown,2].set_title(f"Prediction (Dice={dice:.3f})")
        shown += 1
    if shown >= 4: break
for ax in axes.flat: ax.axis("off")
plt.suptitle("Validation Predictions", fontsize=14); plt.tight_layout()
plt.savefig(OUTPUT_DIR/"val_predictions.png", dpi=100, bbox_inches="tight"); plt.show()
print("Done! Submit /kaggle/working/submission.csv")
'''),
]


def build():
    cells = []
    for kind, source in CELLS:
        lines = source.strip("\n").splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = Path(__file__).resolve().parents[1] / "notebooks" / "nppe1_semantic_segmentation.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
