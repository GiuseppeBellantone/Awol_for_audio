#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
from collections import defaultdict


KEEP_RULES = {
    "brass": {"acoustic"},
    "flute": {"acoustic"},
    "mallet": {"acoustic"},
    "reed": {"acoustic"},
    "string": {"acoustic"},
    "guitar": {"acoustic"},
    "keyboard": {"acoustic"},
    "bass": {"electronic"},
    "organ": {"electronic"},
}

FAMILY_ORDER = ["brass","flute","mallet","reed","string","guitar","keyboard","bass","organ"]

def make_prompt(family: str) -> str:
    # Caption minimalista
    return f"a {family} note"

def load_nsynth_examples(nsynth_root: Path):
    meta = json.loads((nsynth_root / "examples.json").read_text(encoding="utf-8"))
    rows = []
    for note_id, m in meta.items():
        family = m["instrument_family_str"]
        source = m["instrument_source_str"]
        wav_rel = f"nsynth-test/audio/{note_id}.wav"
        rows.append({
            "note_id": note_id,
            "family": family,
            "source": source,
            "pitch_midi": int(m["pitch"]),
            "velocity": int(m["velocity"]),
            "sample_rate": int(m.get("sample_rate", 16000)),
            "wav_path": wav_rel
        })
    return rows

def filter_and_group(rows):
    groups = defaultdict(list)
    for r in rows:
        fam = r["family"]
        src = r["source"]
        if fam in KEEP_RULES and src in KEEP_RULES[fam]:
            groups[(fam, src)].append(r)
    return groups

def balanced_sample(groups, per_combo, seed):
    rnd = random.Random(seed)
    out = []
    combos = sorted(groups.keys(), key=lambda k: (FAMILY_ORDER.index(k[0]) if k[0] in FAMILY_ORDER else 999, k[1]))
    print("Combos kept (family, source, count):")
    for (fam, src) in combos:
        pool = groups[(fam, src)]
        print(f"  - {fam:<10} {src:<10} {len(pool):4d}")
        if len(pool) < per_combo:
            chosen = pool
        else:
            chosen = rnd.sample(pool, per_combo)
        for r in chosen:
            out.append({
                "prompt": make_prompt(fam),
                "wav_path": r["wav_path"],
                "theta_synth": {
                    "pitch_midi": r["pitch_midi"],
                    "velocity":   r["velocity"],
                    "sample_rate": r["sample_rate"]
                }
            })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsynth_root", required=True)
    ap.add_argument("--out", default="data/nsynth/pairs_awol_base.jsonl")
    ap.add_argument("--per_combo", type=int, default=55)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ns_root = Path(args.nsynth_root)
    rows = load_nsynth_examples(ns_root / "nsynth-test" if (ns_root / "nsynth-test").exists() else ns_root)
    groups = filter_and_group(rows)
    picked = balanced_sample(groups, args.per_combo, args.seed)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as fw:
        for rec in picked:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✔ Wrote {len(picked)} clean base records to {outp}")
    print("Each record has: prompt, wav_path, theta_synth{pitch_midi,velocity,sample_rate}")

if __name__ == "__main__":
    main()

