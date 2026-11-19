# Data preparation

Datasets are third-party and are not redistributed. Obtain each one from its provider and follow
its licence. The pipeline has three steps; all of them are resumable.

## 1. Crop faces

```bash
python scripts/prepare_faces.py --input /data/raw/oulu --out /data/faces/oulu \
    --detector-model models/face_detection_yunet_2023mar.onnx --frames-per-video 25
```

Frames are sampled uniformly from every video, detected with YuNet (OpenCV model zoo), rolled so
that the eyes are horizontal, cropped with a 1.3x margin and resized to 224 x 224. Output:
`<out>/<video path without extension>/<frame>.png`. Still images are processed as one-frame videos.

## 2. Build the manifest

```bash
M=data/manifest.csv
python scripts/build_manifest.py --dataset oulu   --root /data/faces/oulu   --out $M --subset-dirs train dev test
python scripts/build_manifest.py --dataset casia  --root /data/faces/casia  --out $M --append
python scripts/build_manifest.py --dataset replay --root /data/faces/replay --out $M --append
python scripts/build_manifest.py --dataset msu    --root /data/faces/msu    --out $M --append
python scripts/build_manifest.py --dataset generic --domain Y --root /data/faces/rose \
    --live-regex genuine --spoof-regex 'print|video|mask' --out $M --append
```

| Domain tag | Dataset | Used by |
|---|---|---|
| `O` `C` `I` `M` | OULU-NPU, CASIA-MFSD, Replay-Attack, MSU-MFSD | OCIM, MICY, LIMITED, OULU_P3/P4 |
| `Y` | ROSE-YOUTU | MICY |
| `CS` `SW` | CelebA-Spoof, SiW-Mv2 | MODERN |
| `WM` | WMCA | WMCA_UNSEEN |

Columns: `path,label,domain,video_id,attack,phone,session,subset`. `label` is 0 live and 1 spoof.
The last four are optional and only needed by the protocols below. If a dataset's folder naming
differs from the built-in adapters (`oulu`, `casia`, `replay`, `msu`, `siw`), use the `generic`
adapter with `--live-regex`, `--spoof-regex` and, for WMCA, `--attack-regex`.

* **OULU_P3 / OULU_P4** need `subset` (`train`, `dev`, `test`), `phone` (1-6), `session` (1-3) and
  `attack` (`print1`, `print2`, `replay1`, `replay2`). The `oulu` adapter derives them from the
  official file names `Phone_Session_User_File` (File 1 live, 2-3 print, 4-5 replay); store the
  crops in `train/`, `dev/`, `test/` folders and pass `--subset-dirs`.
* **WMCA_UNSEEN** needs `attack` for spoof frames. Export WMCA RGB frames to PNG first (the dataset
  ships HDF5), then use the `generic` adapter with `--domain WM --attack-regex 'attack/([^/]+)/'`
  adapted to your folder layout. Each fold holds out one attack type; 20% of bona fide videos are
  held out as test genuine samples (fixed across folds and seeds).

## 3. Cache trace priors

```bash
git clone https://github.com/zllrunning/face-parsing.PyTorch external/face-parsing
# put the pretrained BiSeNet weights at external/face-parsing/res/cp/79999_iter.pth
python scripts/precompute_traces.py --source bisenet --manifest $M \
    --out cache/traces/bisenet --parser-repo external/face-parsing --device cuda
```

Two ablation variants need other attention sources (only for Table 6):

```bash
# Grad-CAM from a trained baseline host
python scripts/precompute_traces.py --source gradcam --manifest $M --out cache/traces/gradcam \
    --checkpoint runs/<baseline_run>/best.pt --backbone vit_convpass
# STDN spoof-trace maps: produce them with the STDN authors' code and save one float map per image
# as cache/traces/stdn/<apr_tma.data.trace_key(path)>.npy
```

Which cache a run reads is controlled by `--trace-root cache/traces` of `scripts/run_suite.py`.
Because the parser is frozen, caching it is equivalent to running it online.
