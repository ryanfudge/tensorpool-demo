from __future__ import annotations
import argparse, time, itertools, math
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights

def parse_args():
    p = argparse.ArgumentParser(description="RegNet-Y-400MF on FGVC-Aircraft")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-vis", action="store_true",
                   help="skip visualisation stage (just train)")
    return p.parse_args()

def get_dataloaders(root: Path, bs: int, workers: int):
    mean,std = [0.485,0.456,0.406],[0.229,0.224,0.225]
    t_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(), transforms.Normalize(mean,std)])
    t_val = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean,std)])

    train_set = datasets.FGVCAircraft(root, split="trainval",
                                      download=True, transform=t_train)
    test_set  = datasets.FGVCAircraft(root, split="test",
                                      download=True, transform=t_val)
    return (
        DataLoader(train_set, bs, shuffle=True,
                   num_workers=workers, pin_memory=True),
        DataLoader(test_set, bs*2, shuffle=False,
                   num_workers=workers, pin_memory=True),
        len(train_set.classes), train_set.classes)

def build_model(num_classes: int, device: str):
    weights = RegNet_Y_400MF_Weights.IMAGENET1K_V2
    m = regnet_y_400mf(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)

def train_one_epoch(model, loader, crit, opt, scaler, dev):
    model.train()
    tloss,correct,total = 0.0,0,0
    for x,y in loader:
        x,y = x.to(dev,non_blocking=True), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(x)
            loss = crit(out,y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

        tloss += loss.item()*x.size(0)
        correct += (out.argmax(1)==y).sum().item()
        total += x.size(0)
    return tloss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, crit, dev, collect_preds=False):
    model.eval()
    vloss,correct,total = 0.0,0,0
    all_preds,all_labels = [],[]
    for x,y in loader:
        x,y = x.to(dev,non_blocking=True), y.to(dev)
        out = model(x)
        vloss += crit(out,y).item()*x.size(0)
        preds = out.argmax(1)
        correct += (preds==y).sum().item()
        total += x.size(0)
        if collect_preds:
            all_preds.append(preds.cpu()); all_labels.append(y.cpu())
    if collect_preds:
        return (vloss/total, correct/total,
                torch.cat(all_preds), torch.cat(all_labels))
    return vloss/total, correct/total

def make_plots(history, class_names, preds, labels, test_loader):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np

    # learning curve
    epochs = range(1, len(history["train_acc"])+1)
    for metric in ("loss","acc"):
        plt.figure()
        plt.plot(epochs, history[f"train_{metric}"], label="train")
        plt.plot(epochs, history[f"val_{metric}"],   label="val")
        plt.xlabel("epoch"); plt.ylabel(metric)
        plt.legend(); plt.grid(True, ls="--", alpha=.4)
        plt.tight_layout(); plt.savefig(f"{metric}_curve.png", dpi=150)

    # confusion matrix 
    cm = confusion_matrix(labels, preds, normalize="true")
    plt.figure(figsize=(7,6))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix (row-norm.)")
    plt.xlabel("predicted"); plt.ylabel("true")
    plt.colorbar(im,fraction=.046)
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)

    # prediction gallery 
    inv_norm = transforms.Normalize(
        mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
        std=[1/s for s in [0.229,0.224,0.225]])
    images, txt = [], []
    # take first 16
    with torch.no_grad():
        for imgs,_ in itertools.islice(test_loader, math.ceil(16/ test_loader.batch_size)):
            images.extend(imgs.cpu())
    images = images[:16]
    for img in images:
        img_orig = inv_norm(img).clamp(0,1)
        images_txt = img_orig

    grid = utils.make_grid([inv_norm(i) for i in images], nrow=4)
    plt.figure(figsize=(8,8)); plt.axis("off"); plt.imshow(grid.permute(1,2,0))
    plt.title("Sample test images")
    plt.tight_layout(); plt.savefig("prediction_gallery.png", dpi=150)

    print(classification_report(labels, preds, target_names=None, digits=3))

def main():
    args = parse_args()
    train_loader, test_loader, n_cls, cls_names = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers)
    model = build_model(n_cls, args.device)

    crit = nn.CrossEntropyLoss()
    opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched= optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler()

    hist = {k:[] for k in
            ("train_loss","train_acc","val_loss","val_acc")}
    best_acc = 0.0
    for epoch in range(1,args.epochs+1):
        t0=time.time()
        tl,ta = train_one_epoch(model,train_loader,crit,opt,scaler,args.device)
        vl,va = evaluate(model,test_loader,crit,args.device)
        sched.step()

        hist["train_loss"].append(tl); hist["train_acc"].append(ta)
        hist["val_loss"].append(vl);   hist["val_acc"].append(va)

        print(f"[{epoch:02}/{args.epochs}] "
              f"train {ta*100:5.2f}% / {tl:.4f}  | "
              f"val {va*100:5.2f}% / {vl:.4f}  | "
              f"lr {sched.get_last_lr()[0]:.2e}  | {time.time()-t0:.1f}s")

        if va>best_acc:
            best_acc=va
            torch.save(model.state_dict(),"best_regnet.pth")

    print(f"✓ training done – best val acc {best_acc*100:.2f}%")

    if not args.no_vis:
        # collect final predictions for visuals
        vl,va,preds,labels = evaluate(model,test_loader,crit,args.device,
                                      collect_preds=True)
        make_plots(hist, cls_names, preds, labels, test_loader)
        try:
            import matplotlib.pyplot as plt; plt.show()
        except Exception:
            pass

if __name__ == "__main__":
    main()