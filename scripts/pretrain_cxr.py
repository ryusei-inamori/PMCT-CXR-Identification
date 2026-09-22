#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import time
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from pmct_cxr_id.model import CNN
from pmct_cxr_id.sam import SAM, enable_running_stats, disable_running_stats


class CXRPatientDataset(Dataset):
    REQUIRED = ["path","label","OriginalImage_Width","OriginalImage_Height","OriginalImagePixelSpacing_x","OriginalImagePixelSpacing_y"]
    def __init__(self, image_root: Path, csv_path: Path, transform):
        self.root = image_root
        self.df = pd.read_csv(csv_path)
        missing = [c for c in self.REQUIRED if c not in self.df]
        if missing: raise ValueError(f"Missing CSV columns: {missing}")
        self.encoder = LabelEncoder().fit(self.df.label)
        self.targets = self.encoder.transform(self.df.label)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        w = max(1, round(float(r.OriginalImage_Width) * float(r.OriginalImagePixelSpacing_x)))
        h = max(1, round(float(r.OriginalImage_Height) * float(r.OriginalImagePixelSpacing_y)))
        image = Image.open(self.root / r.path).convert("L").resize((w,h), Image.Resampling.BICUBIC)
        return self.transform(image), int(self.targets[i])


def main():
    p=argparse.ArgumentParser(description="ChestXray14 patient-ID pretraining")
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--dataset-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=1280)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--eta-min", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=2.0)
    p.add_argument("--momentum", type=float, default=0.8)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--imagenet-init", action=argparse.BooleanOptionalAction, default=True)
    a=p.parse_args()

    torch.manual_seed(a.seed)
    device=a.device if (a.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    tfm=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomPerspective(distortion_scale=0.25,p=0.1,interpolation=transforms.InterpolationMode.BICUBIC,fill=0),
        transforms.RandomRotation(10),
        transforms.Resize((320,320),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    ds=CXRPatientDataset(a.image_root,a.dataset_csv,tfm)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,pin_memory=True)
    n_classes=len(ds.encoder.classes_)
    print(f"images={len(ds)} classes={n_classes}")
    model=CNN(imagenet_init=a.imagenet_init,n_classes=n_classes,hidden_dim=a.hidden_dim,dropout=a.dropout).to(device)
    criterion=nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    opt=SAM(model.parameters(),torch.optim.SGD,rho=a.rho,adaptive=True,lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay,nesterov=False)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer,T_max=a.epochs,eta_min=a.eta_min)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    log=[]
    for epoch in range(a.epochs):
        model.train(); total=correct=0; loss_sum=0.; t0=time.time()
        for x,y in loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            enable_running_stats(model)
            logits=model(x,y); loss=criterion(logits,y); loss.backward(); opt.first_step(zero_grad=True)
            disable_running_stats(model)
            criterion(model(x,y),y).backward(); opt.second_step(zero_grad=True)
            with torch.no_grad():
                total += y.numel(); correct += (logits.argmax(1)==y).sum().item(); loss_sum += loss.item()*y.numel()
        sched.step()
        row={"epoch":epoch,"lr":opt.base_optimizer.param_groups[0]["lr"],"loss":loss_sum/total,"accuracy":correct/total,"seconds":time.time()-t0}
        log.append(row); print(row)
        pd.DataFrame(log).to_csv(a.output_dir/"log.csv",index=False)
    torch.save({"epoch":a.epochs-1,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict()},a.output_dir/"finalmodel.cpt")

if __name__ == "__main__": main()
