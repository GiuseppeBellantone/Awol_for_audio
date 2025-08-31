#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, ClapTextModelWithProjection

THETA_KEYS = [
    "pitch_midi","velocity","sample_rate","duration_s",
    "attack_s","decay_s","sustain","release_s",
    "mod_ratio","mod_index",
]

def load_text_encoder(device="cpu"):
    name = "laion/clap-htsat-fused"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = ClapTextModelWithProjection.from_pretrained(name)
    mdl.to(device).eval()
    return tok, mdl

def encode_texts(tokenizer, model, prompts, device="cpu", batch_size=64):
    embs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**inputs)              # out.text_embeds
            emb = torch.nn.functional.normalize(out.text_embeds, p=2, dim=1)  # L2 norm
            embs.append(emb.cpu())
    return torch.cat(embs, dim=0).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="data/nsynth/pairs_awol_fm.jsonl")
    ap.add_argument("--out_jsonl", default="data/nsynth/pairs_awol_fm_clap.jsonl")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    use_device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"

    in_path = Path(args.in_jsonl)
    recs = []
    prompts = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            recs.append(rec)
            prompts.append(str(rec.get("prompt","")))

    tok, mdl = load_text_encoder(use_device)
    text_emb = encode_texts(tok, mdl, prompts, device=use_device, batch_size=args.batch_size)
    assert text_emb.shape[0] == len(recs)

    out_path = Path(args.out_jsonl); out_path.parent.mkdir(parents=True, exist_ok=True)
    n=0
    with out_path.open("w", encoding="utf-8") as fw:
        for rec, emb in zip(recs, text_emb):
            theta_vec = rec.get("theta_vec", None)
            if theta_vec is None:
                th = rec.get("theta_synth", {})
                theta_vec = [float(th[k]) for k in THETA_KEYS]
                rec["theta_vec"] = theta_vec
                rec["theta_keys"] = THETA_KEYS

            out = dict(rec)
            out["text_emb"] = emb.astype(float).tolist()
            fw.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f" Wrote {n} pairs to {out_path}")
    print(f"text_emb dim = {text_emb.shape[1]}, theta dim = {len(recs[0]['theta_vec'])}")

if __name__ == "__main__":
    main()
