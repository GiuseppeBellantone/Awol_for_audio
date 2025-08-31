# Text-to-Audio Synthesis with Conditional Real-NVP

This project implements a complete pipeline to generate **audio synthesis parameters** from textual descriptions (captions).  
The goal is to map text embeddings obtained with **CLAP** (Contrastive Language-Audio Pretraining) into the parameter space of a simplified synthesizer, and then reconstruct an audio.

---

## Project Overview

The workflow is divided into multiple steps, each handled by a dedicated script:

1. **Build caption–audio pairs**
   - Script: `scripts/build_pairs.py`  
   - Filters and balances examples from the Nsynth-test dataset (fraction of the larger Nsynth dataset)  
   - Produces minimal captions (e.g., `"a guitar note"`) linked to `.wav`.

2. **Extract synthesis parameters (ADSR + FM)**
   - Script: `scripts/extract_params.py`  
   - Analyzes audio to estimate:
     - Pitch, velocity, sample rate (already known from nsynth) 
     - ADSR envelope (attack, decay, sustain, release)  
     - FM ratio and modulation index  
   - Stores parameters in a `theta_vec`.

3. **Generate text embeddings**
   - Script: `scripts/make_text_emb.py`  
   - Uses **CLAP (laion/clap-htsat-fused)** to encode captions into normalized embeddings.  
   - Outputs a JSONL dataset with pairs `(text_emb, theta_vec)`.

4. **Train the flow model**
   - Script: `scripts/train_flow.py`  
   - Trains a **Conditional Real-NVP** network that maps text embeddings to `theta_vec`.  
   - Loss function includes:
     - L<sub>rec</sub> (reconstruction)
     - L<sub>cyc</sub> (cycle consistency)
     - L<sub>z</sub> (prior regularization)

5. **Inference: generate audio from caption**
   - Script: `scripts/synthesize.py`  
   - Pipeline:
     ```
     Caption → CLAP embedding → Flow model → Theta parameters → Synthesizer → .wav file
     ```

---


### The present model (checkpoints/model.pt) was created executing:

   - python scripts\build_pairs.py --nsynth_root . --out data\nsynth\pairs_awol_base.jsonl --per_combo 55 --seed 123
   - python scripts\extract_params.py --in_jsonl data\nsynth\pairs_awol_base.jsonl --out_jsonl data\nsynth\pairs_awol_fm.jsonl --root .  
   - python scripts\make_text_emb.py --in_jsonl data\nsynth\pairs_awol_fm.jsonl --out_jsonl data\nsynth\pairs_awol_fm_clap.jsonl --device cuda
   - python scripts/train_flow.py --data data/nsynth/pairs_awol_fm_clap.jsonl --out checkpoints/model.pt --device cuda --epochs 3500 --batch_size 128 --layers 10 --hidden 768 --dropout 0.05 --learned_masks --w_rec 1.0 --w_cyc 0.5 --w_prior 0.15 --huber_beta 0.02 --text_noise 0.01 --lr 3e-4

   You can use different paramters and train a new model (number of layers, batch size, etc.)

### Example of generation of  a wav file from a caption:

python scripts/synthesize.py  --ckpt checkpoints/model.pt  --prompt "a guitar note" --dump_json

The script will generate a .wav file renders/a_guitar_note.wav


### Requirements


Python 3.11.9 was used for the project


```bash
pip install -r requirements.txt
