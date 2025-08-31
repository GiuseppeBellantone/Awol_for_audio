#!/usr/bin/env python3
import argparse, json, math, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class ClapThetaDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.recs = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                te = np.array(obj["theta_vec"], dtype=np.float32)
                tx = np.array(obj["text_emb"], dtype=np.float32)
                self.recs.append((tx, te))
        self.text_dim = self.recs[0][0].shape[0]
        self.theta_dim = self.recs[0][1].shape[0]

    def __len__(self): return len(self.recs)

    def __getitem__(self, idx):
        tx, te = self.recs[idx]
        return torch.from_numpy(tx), torch.from_numpy(te)


@torch.no_grad()
def compute_scaler_from_indices(ds: ClapThetaDataset, indices):
    thetas = torch.stack([torch.from_numpy(ds.recs[i][1]) for i in indices]).float()
    txs    = torch.stack([torch.from_numpy(ds.recs[i][0]) for i in indices]).float()
    th_mean, th_std = thetas.mean(0), thetas.std(0)
    th_std[th_std < 1e-6] = 1.0
    tx_mean, tx_std = txs.mean(0), txs.std(0)
    tx_std[tx_std < 1e-6] = 1.0
    return (th_mean, th_std), (tx_mean, tx_std)


class ResidualMLP(nn.Module):
    def __init__(self, inp, hid, out, pdrop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, out)
        self.act = nn.ReLU()
        self.do  = nn.Dropout(pdrop)
        self.ln1 = nn.LayerNorm(hid)
        self.ln2 = nn.LayerNorm(hid)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(self.ln1(h))
        h = self.do(h)
        h = self.fc2(h)
        h = self.act(self.ln2(h))
        h = self.do(h)
        return self.fc3(h)

class CouplingLayer(nn.Module):
    def __init__(self, D, C, hidden=512, learn_mask=False, pdrop=0.0):
        super().__init__()
        self.D, self.C = D, C
        self.learn_mask = learn_mask
        if learn_mask:
            alt = torch.arange(D) % 2
            init = alt.float()
            self.mask_logits = nn.Parameter(torch.logit(torch.clamp(init, 1e-4, 1-1e-4)))
        else:
            self.register_buffer("fixed_mask", (torch.arange(D) % 2).float())

        self.c_ln = nn.LayerNorm(C)
        self.st_net = ResidualMLP(inp=D + C, hid=hidden, out=2*D, pdrop=pdrop)
        self.s_scale = nn.Parameter(torch.tensor(1.0))

    def mask(self):
        if self.learn_mask:
            return torch.sigmoid(self.mask_logits)
        else:
            return self.fixed_mask

    def forward(self, x, c, reverse=False):
        m = self.mask()
        m_b = m.unsqueeze(0).expand_as(x)
        c = self.c_ln(c)
        xc = torch.cat([x * m_b, c], dim=-1)
        st = self.st_net(xc)
        s, t = torch.chunk(st, 2, dim=-1)
        s = torch.tanh(s) * self.s_scale

        if not reverse:
            y = m_b * x + (1 - m_b) * (x * torch.exp(s) + t)
            return y
        else:
            y = ((x - t) * torch.exp(-s)) * (1 - m_b) + m_b * x
            return y

class ConditionalRealNVP(nn.Module):
    def __init__(self, theta_dim, text_dim, num_layers=5, hidden=512, learned_masks=False, pdrop=0.0):
        super().__init__()
        self.D, self.C = theta_dim, text_dim
        self.layers = nn.ModuleList([
            CouplingLayer(self.D, self.C, hidden=hidden, learn_mask=learned_masks, pdrop=pdrop)
            for _ in range(num_layers)
        ])
        perms = [torch.randperm(self.D) for _ in range(num_layers)]
        invs  = [torch.argsort(p) for p in perms]
        self.register_buffer("perms", torch.stack(perms, 0))
        self.register_buffer("invs",  torch.stack(invs,  0))

    def f(self, x, c):
        z = x
        for i, layer in enumerate(self.layers):
            p = self.perms[i]
            z = z[..., p]
            z = layer(z, c, reverse=False)
        return z

    def g(self, z, c):
        x = z
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            x = layer(x, c, reverse=True)
            inv = self.invs[i]
            x = x[..., inv]
        return x

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v)

def train(args):
    device = torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    data_path = Path(args.data)
    ds = ClapThetaDataset(data_path)

    n_total = len(ds)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_indices = train_ds.indices if hasattr(train_ds, "indices") else list(range(n_train))


    (theta_mean, theta_std), (text_mean, text_std) = compute_scaler_from_indices(ds, train_indices)
    theta_mean, theta_std = theta_mean.to(device), theta_std.to(device)
    text_mean,  text_std  = text_mean.to(device),  text_std.to(device)

    def collate(batch):
        tx = torch.stack([b[0] for b in batch]).float()
        th = torch.stack([b[1] for b in batch]).float()
        thn = (th - theta_mean.cpu()) / theta_std.cpu()
        txn = (tx - text_mean.cpu()) / text_std.cpu()
        return txn, thn, th  

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    model = ConditionalRealNVP(theta_dim=ds.theta_dim, text_dim=ds.text_dim,
                               num_layers=args.layers, hidden=args.hidden,
                               learned_masks=args.learned_masks, pdrop=args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=30, min_lr=args.lr_min, verbose=True)
    ema = EMA(model, decay=args.ema)

    recon_loss = nn.SmoothL1Loss(beta=args.huber_beta) 
    best_val = float("inf")

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_rec, tr_cyc, tr_pri, tr_tot = 0., 0., 0., 0.
        for txn, thn, _ in train_loader:
            txn, thn = txn.to(device), thn.to(device)

            if args.text_noise > 0:
                txn = txn + torch.randn_like(txn) * args.text_noise

            z0 = torch.zeros_like(thn)
            pred_thn = model.g(z0, txn)
            loss_rec = recon_loss(pred_thn, thn)

            z_hat = model.f(thn, txn)
            thn_rec2 = model.g(z_hat, txn)
            loss_cyc = recon_loss(thn_rec2, thn)
            loss_prior = torch.mean(z_hat**2)

            loss = args.w_rec * loss_rec + args.w_cyc * loss_cyc + args.w_prior * loss_prior

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            B = txn.size(0)
            tr_rec  += loss_rec.item()  * B
            tr_cyc  += loss_cyc.item()  * B
            tr_pri  += loss_prior.item()* B
            tr_tot  += loss.item()      * B

        Ntr = len(train_loader.dataset)
        tr_rec /= Ntr; tr_cyc /= Ntr; tr_pri /= Ntr; tr_tot /= Ntr

        model.eval()
        val_rec = val_cyc = val_pri = 0.0
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

        with torch.no_grad():
            for txn, thn, th_raw in val_loader:
                txn, thn = txn.to(device), thn.to(device)

                z0 = torch.zeros_like(thn)
                pred_thn = model.g(z0, txn)
                z_hat = model.f(thn, txn)
                thn_rec2 = model.g(z_hat, txn)

                loss_rec = recon_loss(pred_thn, thn)
                loss_cy  = recon_loss(thn_rec2, thn)
                loss_pr  = torch.mean(z_hat**2)

                B = txn.size(0)
                val_rec += loss_rec.item() * B
                val_cyc += loss_cy.item()  * B
                val_pri += loss_pr.item()  * B

        Nva = len(val_loader.dataset)
        val_rec /= Nva; val_cyc /= Nva; val_pri /= Nva
        val_tot = args.w_rec*val_rec + args.w_cyc*val_cyc + args.w_prior*val_pri

        model.load_state_dict(backup)

        sched.step(val_tot)

        print(f"Epoch {epoch:03d} | "
              f"train Lrec {tr_rec:.4f} Lcyc {tr_cyc:.4f} Lz {tr_pri:.4f} | "
              f"val Lrec {val_rec:.4f} Lcyc {val_cyc:.4f} Lz {val_pri:.4f} | tot {val_tot:.4f}")

        if val_tot < best_val:
            best_val = val_tot
            ckpt = {
                "state_dict": model.state_dict(),
                "theta_mean": theta_mean.detach().cpu().numpy().tolist(),
                "theta_std":  theta_std.detach().cpu().numpy().tolist(),
                "text_mean":  text_mean.detach().cpu().numpy().tolist(),
                "text_std":   text_std.detach().cpu().numpy().tolist(),
                "theta_dim": ds.theta_dim,
                "text_dim":  ds.text_dim,
                "config": {
                    "layers": args.layers,
                    "hidden": args.hidden,
                    "learned_masks": args.learned_masks,
                    "dropout": args.dropout
                }
            }
            outp = Path(args.out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, outp)
            print(f"  -> saved best to {outp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/nsynth/pairs_awol_fm_clap.jsonl")
    ap.add_argument("--out",  default="checkpoints/flow_awol_fm.pt")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--epochs", type=int, default=400)              
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_min", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--learned_masks", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--w_rec", type=float, default=1.0)
    ap.add_argument("--w_cyc", type=float, default=0.5)
    ap.add_argument("--w_prior", type=float, default=0.1)
    ap.add_argument("--huber_beta", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--text_noise", type=float, default=0.01)

    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    train(args)

if __name__ == "__main__":
    main()
