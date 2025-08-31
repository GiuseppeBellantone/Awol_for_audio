#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt, find_peaks

THETA_KEYS = [
    "pitch_midi","velocity","sample_rate","duration_s",
    "attack_s","decay_s","sustain","release_s",
    "mod_ratio","mod_index",
]

def clamp(x, lo, hi): return float(max(lo, min(hi, x)))

def midi_to_hz(m): return 440.0 * (2.0 ** ((m - 69) / 12.0))

def butter_highpass(y, sr, cutoff_hz=30.0, order=2):
    if cutoff_hz<=0: return y
    b,a = butter(order, cutoff_hz/(0.5*sr), btype="highpass")
    return filtfilt(b,a,y)

def analytic_envelope(y, sr, smooth_hz=20.0):
    from scipy.signal import hilbert
    env = np.abs(hilbert(y))
    if smooth_hz and smooth_hz>0:
        b,a=butter(2, smooth_hz/(0.5*sr), btype="lowpass")
        env=filtfilt(b,a,env)
    return np.maximum(env,1e-9)

def robust_sustain_window(n, head=0.3, tail=0.3):
    i0=int(round(n*head)); i1=max(i0+1, int(round(n*(1.0-tail))))
    return i0,i1

def estimate_adsr(env, sr):
    n=len(env); dur=n/sr
    peak=float(np.max(env))
    if peak<=0:  
        return 0.01,0.05,0.7,0.05,dur
    e=env/peak
    idx10=np.where(e>=0.1)[0]; idx90=np.where(e>=0.9)[0]
    attack = (idx90[0]-idx10[0])/sr if len(idx10)>0 and len(idx90)>0 else 0.01
    s0,s1=robust_sustain_window(n,0.3,0.3)
    sustain=float(np.median(e[s0:s1])) if s1>s0 else 0.7
    sustain=clamp(sustain,0.0,1.0)
    i_peak=int(np.argmax(e)); below=np.where(e[i_peak:]<=sustain+0.02)[0]
    decay=(below[0]/sr) if len(below)>0 else 0.05
    r0=int(0.8*n); tail=e[r0:]; below5=np.where(tail<=0.05)[0]
    release=(below5[0]/sr) if len(below5)>0 else 0.05
    return float(attack), float(decay), float(sustain), float(release), float(dur)

def estimate_fm(y, sr, fc_hz):
    n=len(y)
    if n<sr//10 or fc_hz<=0: return 1.0, 0.5
    s0,s1=robust_sustain_window(n,0.3,0.3)
    seg=y[s0:s1] if s1>s0 else y
    if len(seg)<512: seg=y
    w=np.hanning(len(seg))
    X=np.fft.rfft(seg*w)
    mag=np.abs(X)+1e-12
    fre=np.fft.rfftfreq(len(seg),1.0/sr)
    bw=2000.0
    mask=(fre>=max(0.0,fc_hz-bw))&(fre<=fc_hz+bw)
    if not np.any(mask): return 1.0,0.5
    mwin=mag[mask]; fwin=fre[mask]
    if len(mwin)<16: return 1.0,0.5
    peaks,_=find_peaks(mwin, prominence=np.max(mwin)*0.1)
    if len(peaks)>=2:
        diffs=np.diff(fwin[peaks]); diffs=diffs[diffs>1.0]
        fm_hz=float(np.median(diffs)) if len(diffs)>0 else fc_hz
    else:
        spec=mwin/np.max(mwin)
        ac=np.correlate(spec,spec,mode="full")[len(spec)-1:]
        pk,_=find_peaks(ac, distance=3)
        if len(pk)>=2:
            bin_spacing=np.median(np.diff(pk[:5]))
            fm_hz=(bin_spacing/max(1,len(fwin)-1))*(fwin[-1]-fwin[0])
        else:
            fm_hz=fc_hz
    mod_ratio=clamp(fm_hz/max(1e-6,fc_hz),0.25,8.0)
    thr=float(np.max(mwin))*(10**(-30.0/20.0))
    idx=np.where(mwin>=thr)[0]
    BW=float(fwin[idx[-1]]-fwin[idx[0]]) if len(idx)>0 else 0.0
    if fm_hz<=1e-3: mod_index=0.5
    else:
        I=BW/(2.0*fm_hz)-1.0
        mod_index=clamp(I,0.0,10.0)
    return float(mod_ratio), float(mod_index)

def process_record(rec, root_dir: Path):
    wav_path = Path(rec["wav_path"])
    if not wav_path.is_absolute():
        wav_path = (root_dir / wav_path).resolve()
    y,sr = sf.read(wav_path.as_posix())
    if y.ndim>1: y=y[:,0].astype(np.float32)
    else: y=y.astype(np.float32)
    y = butter_highpass(y, sr, 30.0, 2)

    th = dict(rec["theta_synth"]) 
    th["duration_s"] = max(0.05, len(y)/sr)
    env = analytic_envelope(y, sr, 20.0)
    a,d,s,r,dur = estimate_adsr(env, sr)
    th["attack_s"]  = clamp(a, 0.0, 0.2)
    th["decay_s"]   = clamp(d, 0.0, 0.4)
    th["sustain"]   = clamp(s, 0.0, 1.0)
    th["release_s"] = max(0.02, clamp(r, 0.0, 0.6))
    fc_hz = midi_to_hz(th["pitch_midi"])
    ratio, index = estimate_fm(y, sr, fc_hz)
    th["mod_ratio"] = ratio
    th["mod_index"] = index
    theta_vec = [float(th[k]) for k in THETA_KEYS]
    out = dict(rec)
    out["theta_synth"] = th
    out["theta_vec"] = theta_vec
    out["theta_keys"] = THETA_KEYS
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="data/nsynth/pairs_awol_base.jsonl")
    ap.add_argument("--out_jsonl", default="data/nsynth/pairs_awol_fm.jsonl")
    ap.add_argument("--root", default=".", help="root per risolvere wav_path")
    args = ap.parse_args()

    inp=Path(args.in_jsonl); outp=Path(args.out_jsonl); outp.parent.mkdir(parents=True, exist_ok=True)
    root_dir=Path(args.root)
    n=0; skipped=0
    with inp.open("r", encoding="utf-8") as fr, outp.open("w", encoding="utf-8") as fw:
        for line in fr:
            if not line.strip(): continue
            rec=json.loads(line)
            try:
                rec2=process_record(rec, root_dir)
                fw.write(json.dumps(rec2, ensure_ascii=False)+"\n")
                n+=1
            except Exception as e:
                print(f"[WARN] skip {rec.get('wav_path')} -> {e}")
                skipped+=1
    print(f" Wrote {n} records to {outp} (skipped {skipped})")
    print(f"Each record now has theta_synth with ADSR+FM and theta_vec in order: {THETA_KEYS}")

if __name__ == "__main__":
    main()
