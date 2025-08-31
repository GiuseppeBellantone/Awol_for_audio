#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from transformers import AutoTokenizer, ClapTextModelWithProjection

THETA_KEYS = [
    "pitch_midi","velocity","sample_rate","duration_s",
    "attack_s","decay_s","sustain","release_s",
    "mod_ratio","mod_index",
]

def clamp(x, lo, hi): return float(max(lo, min(hi, x)))
def midi_to_hz(m): return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def load_text_encoder(device="cpu"):
    name = "laion/clap-htsat-fused"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = ClapTextModelWithProjection.from_pretrained(name)
    mdl.to(device).eval()
    return tok, mdl

@torch.no_grad()
def encode_texts(tokenizer, model, prompts, device="cpu", batch_size=64):
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**inputs)  # .text_embeds
        emb = torch.nn.functional.normalize(out.text_embeds, p=2, dim=1)
        outs.append(emb)
    return torch.cat(outs, dim=0)

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
        h = self.fc1(x); h = self.act(self.ln1(h)); h = self.do(h)
        h = self.fc2(h); h = self.act(self.ln2(h)); h = self.do(h)
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
        return torch.sigmoid(self.mask_logits) if self.learn_mask else self.fixed_mask
    def forward(self, x, c, reverse=False):
        m = self.mask(); m_b = m.unsqueeze(0).expand_as(x)
        c = self.c_ln(c)
        st = self.st_net(torch.cat([x * m_b, c], dim=-1))
        s, t = torch.chunk(st, 2, dim=-1)
        s = torch.tanh(s) * self.s_scale
        if not reverse:
            return m_b * x + (1 - m_b) * (x * torch.exp(s) + t)
        else:
            return ((x - t) * torch.exp(-s)) * (1 - m_b) + m_b * x

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
            z = z[..., self.perms[i]]
            z = layer(z, c, reverse=False)
        return z
    def g(self, z, c):
        x = z
        for i in reversed(range(len(self.layers))):
            x = self.layers[i](x, c, reverse=True)
            x = x[..., self.invs[i]]
        return x

def load_flow(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]
    model = ConditionalRealNVP(
        theta_dim=ckpt["theta_dim"],
        text_dim=ckpt["text_dim"],
        num_layers=cfg.get("layers", 6),
        hidden=cfg.get("hidden", 768),
        learned_masks=cfg.get("learned_masks", False),
        pdrop=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    stats = {
        "theta_mean": torch.tensor(ckpt["theta_mean"], dtype=torch.float32, device=device),
        "theta_std":  torch.tensor(ckpt["theta_std"],  dtype=torch.float32, device=device),
        "text_mean":  torch.tensor(ckpt["text_mean"],  dtype=torch.float32, device=device),
        "text_std":   torch.tensor(ckpt["text_std"],   dtype=torch.float32, device=device),
    }
    return model, stats

def make_adsr_envelope(n, sr, a, d, s, r, dur):
    n_total = int(round(dur*sr))
    n_total = max(8, min(n_total, n))
    nA = int(round(a*sr)); nD = int(round(d*sr)); nR = int(round(r*sr))
    nS = max(0, n_total - (nA + nD + nR))
    env = np.zeros(n_total, dtype=np.float32)
    if nA > 0: env[:nA] = np.linspace(0.0, 1.0, nA, endpoint=False, dtype=np.float32)
    i = nA
    if nD > 0: env[i:i+nD] = np.linspace(1.0, s, nD, endpoint=False, dtype=np.float32); i += nD
    if nS > 0: env[i:i+nS] = s; i += nS
    nR = min(nR, n_total - i)
    if nR > 0:
        start = env[i-1] if i>0 else s
        env[i:i+nR] = np.linspace(start, 0.0, nR, endpoint=True, dtype=np.float32)
    return env

def synth_fm(theta):
    pitch_midi = clamp(theta.get("pitch_midi", 60.0), 0.0, 127.0)
    velocity   = clamp(theta.get("velocity",   100.0), 0.0, 127.0)
    sr         = int(clamp(theta.get("sample_rate", 16000.0), 8000.0, 48000.0))
    dur        = clamp(theta.get("duration_s", 1.0), 0.05, 10.0)
    a = clamp(theta.get("attack_s",  0.01), 0.0, 1.0)
    d = clamp(theta.get("decay_s",   0.05), 0.0, 2.0)
    s = clamp(theta.get("sustain",   0.7),  0.0, 1.0)
    r = clamp(theta.get("release_s", 0.1),  0.01, 3.0)
    ratio = clamp(theta.get("mod_ratio", 1.0), 0.25, 8.0)
    index = clamp(theta.get("mod_index", 0.5), 0.0, 10.0)

    n = int(round(dur*sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    fc = midi_to_hz(pitch_midi); fm = ratio * fc
    phase = 2.0*np.pi*fc*t + index * np.sin(2.0*np.pi*fm*t, dtype=np.float32)
    y = np.sin(phase, dtype=np.float32)

    env = make_adsr_envelope(n, sr, a, d, s, r, dur)
    env = np.pad(env, (0, max(0, n - len(env))), mode="edge")
    amp = (velocity / 127.0) ** 1.5
    y = y * env * amp
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 0.99: y = 0.99 * y / peak
    return y.astype(np.float32), sr

@torch.no_grad()
def predict_theta_vectors(prompts, ckpt_path, device="cpu", z_temp=0.0, batch_size=32):
    model, stats = load_flow(ckpt_path, device)
    tok, mdl = load_text_encoder(device)
    tx = encode_texts(tok, mdl, prompts, device=device, batch_size=batch_size)
    txn = (tx - stats["text_mean"]) / stats["text_std"]
    B = txn.size(0)
    if z_temp <= 0.0:
        z = torch.zeros(B, model.D, device=device)
    else:
        z = torch.randn(B, model.D, device=device) * float(z_temp)
    thn = model.g(z, txn)
    th  = thn * stats["theta_std"] + stats["theta_mean"]
    return th.cpu().numpy()

def vec_to_dict(vec):
    return {k: float(v) for k, v in zip(THETA_KEYS, vec.tolist())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--prompt", required=True, help='Es.: "a brass note"')
    ap.add_argument("--out_wav", default=None, help="Percorso WAV in output (default: auto)")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--z_temp", type=float, default=0.0, help="0.0=deterministico (z=0), >0 variazioni")
    ap.add_argument("--dump_json", action="store_true", help="Salva anche i parametri previsti")
    args = ap.parse_args()

    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    prompts = [args.prompt]
    thetas = predict_theta_vectors(prompts, args.ckpt, device=device, z_temp=args.z_temp)
    theta = vec_to_dict(thetas[0])

    y, sr = synth_fm(theta)

    out_dir = Path("renders"); out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join([c if (c.isalnum() or c in "._-") else "_" for c in args.prompt])[:80]
    wav_path = Path(args.out_wav) if args.out_wav else (out_dir / f"{safe or 'sample'}.wav")
    sf.write(wav_path.as_posix(), y, sr)
    print(f"✔ Saved audio to {wav_path}  (sr={sr}, dur={len(y)/sr:.2f}s)")
    if args.dump_json:
        meta = {"prompt": args.prompt, "theta": theta, "wav": str(wav_path)}
        (wav_path.with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  + params: {wav_path.with_suffix('.json')}")

if __name__ == "__main__":
    main()
