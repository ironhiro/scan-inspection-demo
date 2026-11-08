"""Train a small U-Net for per-pixel damage segmentation and export to ONNX.

Input : RGB float32 [1,3,624,448], value/255 - 0.5
Output: cls  int32  [1,624,448]  argmax class
        conf float32 [1,624,448] softmax prob of that class
"""
import glob, json, os, sys, time
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import CLASS_NAMES, MODEL_H, MODEL_W

NC = len(CLASS_NAMES)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128, 192)):
        super().__init__()
        self.enc = nn.ModuleList()
        prev = 3
        for c in ch:
            self.enc.append(block(prev, c))
            prev = c
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.dec.append(block(c * 2, c))
            prev = c
        self.head = nn.Conv2d(prev, NC, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)


class Export(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        p = torch.softmax(self.net(x), 1)
        conf, cls = p.max(1)
        return cls.to(torch.int32), conf


class Cards(Dataset):
    def __init__(self, split, aug):
        self.files = sorted(f for f in glob.glob(os.path.join(DATA, split, "*.jpg")))
        self.aug = aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        m = cv2.imread(f[:-4] + "_mask.png", 0)
        if self.aug:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if rng.random() < 0.5:
                img, m = img[::-1], m[::-1]
            img = img[..., rng.permutation(3)]
            img = img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
            img += rng.normal(0, rng.uniform(0, 4), img.shape)
            img = img.clip(0, 255)
        x = torch.from_numpy(np.ascontiguousarray(img, np.float32)).permute(2, 0, 1) / 255 - 0.5
        return x, torch.from_numpy(np.ascontiguousarray(m)).long()


def dice_loss(logits, y):
    """soft Dice over damage classes that actually occur in the batch; classes
    absent from the batch would otherwise push their predictions to zero"""
    p = torch.softmax(logits.float(), 1)[:, 1:]
    t = F.one_hot(y, NC).permute(0, 3, 1, 2)[:, 1:].float()
    inter = (p * t).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    if not present.any():
        return logits.new_zeros(())
    return (1 - (2 * inter + 1) / (den + 1))[present].mean()


@torch.no_grad()
def evaluate(net, loader, dev):
    net.eval()
    inter = torch.zeros(NC, device=dev)
    union = torch.zeros(NC, device=dev)
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            pr = net(x).argmax(1)
        for c in range(NC):
            a, b = pr == c, y == c
            inter[c] += (a & b).sum()
            union[c] += (a | b).sum()
    net.train()
    return (inter / union.clamp(min=1)).tolist()


def main(epochs=30, bs=8):
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    tr = DataLoader(Cards("train", True), bs, shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)
    va = DataLoader(Cards("val", False), bs, num_workers=4)
    net = UNet().to(dev)
    print("params", sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * len(tr), pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda')
    w = torch.tensor([0.3, 1.0, 1.0, 1.5, 2.0], device=dev)
    best, log = -1, []
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(x)
                loss = F.cross_entropy(out, y, weight=w) + dice_loss(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        iou = evaluate(net, va, dev)
        miou = float(np.mean(iou[1:]))
        log.append({"epoch": ep + 1, "loss": tot / len(tr), "val_iou": iou, "val_miou_fg": miou})
        print(f"ep {ep+1:02d} loss {tot/len(tr):.4f} fg-mIoU {miou:.3f} "
              + " ".join(f"{CLASS_NAMES[c]}={iou[c]:.3f}" for c in range(1, NC)),
              f"{time.time()-t0:.0f}s", flush=True)
        if miou > best:
            best = miou
            torch.save(net.state_dict(), os.path.join(OUT, "unet.pt"))
    with open(os.path.join(OUT, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)

    net.load_state_dict(torch.load(os.path.join(OUT, "unet.pt")))
    net = net.float().cpu().eval()
    torch.onnx.export(Export(net), torch.zeros(1, 3, MODEL_H, MODEL_W), os.path.join(OUT, "damage_unet.onnx"),
                      input_names=["image"], output_names=["cls", "conf"], opset_version=17)
    print("exported, best fg-mIoU", best)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
