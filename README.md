# APR-TMA

Reference implementation of **Acquisition Pipeline Randomization with Trace-Modulated Attention
(APR-TMA)**, a training-time augmentation for domain-generalizable face anti-spoofing, from the
manuscript *Image-Formation Asymmetry: Acquisition Pipeline Randomization with Trace-Modulated
Attention for Domain-Generalizable Face Anti-Spoofing*.

A live face is imaged once; a spoof is captured, reproduced on an attack medium, and captured
again. APR-TMA passes live faces through **one** randomized camera pipeline and spoofs through
**two** coupled pipelines joined by a reproduction layer, while keeping the perturbation strength
**identical across classes**, so that acquisition depth, not magnitude, carries the label. A
label-free face-parsing prior localizes the perturbation. The augmentation is used only during
training: the deployed model is its backbone, with no added parameters or inference cost.

> **Status.** This repository is a clean re-implementation written from the method description
> (Section III, Table 2, Algorithm 1) of the manuscript. Please read [Fidelity notes](#fidelity-notes)
> before comparing numbers with the paper.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                        # 43 tests; no GPU or data needed
```

## Workflow

```
raw datasets ──prepare_faces.py──▶ face crops ──build_manifest.py──▶ manifest.csv
                                         └─precompute_traces.py──▶ cached trace priors
manifest.csv ──run_suite.py──▶ runs/*/result.json ──make_tables.py / statistics.py──▶ tables/
```

1. **Prepare data** ([docs/DATA.md](docs/DATA.md)): crop faces, build one CSV manifest, cache the
   BiSeNet trace priors.
2. **Run experiments** ([docs/REPRODUCING.md](docs/REPRODUCING.md)):
   `python scripts/run_suite.py all --manifest data/manifest.csv --trace-root cache/traces`
   expands `configs/suites.yaml` into jobs (target x method/variant x seed), skips finished runs
   and supports `--shard`, `--targets`, `--seeds`, `--dry-run`.
3. **Build the results**: `make tables stats efficiency` (or the scripts directly).

A single run: `python train.py --manifest data/manifest.csv --protocol OCIM --target M
--backbone vit_convpass --method apr_tma --trace-dir cache/traces/bisenet --out runs/demo`.

## What is implemented

| Manuscript item | Code |
|---|---|
| Eq. (3)-(4) acquisition depth, one pass live / two passes spoof | `pipeline.acquisition_depth`, `pipeline.apply_pipeline` |
| Table 2 stages and distributions | `pipeline.PipelineParams.sample`, `pipeline.acquisition_pass` |
| Eq. (5) inter-pass coupling | `pipeline.couple_params` |
| Reproduction layer `R_xi` | `pipeline.reproduce` |
| Eq. (6)-(7) modulation and blend | `tma.modulation_map`, `tma.blend` |
| Gate, operating point, class-symmetric alpha | `augment.APRTMA`, `augment.APRTMAConfig` |
| Eq. (8)-(9) objective with SARE | `losses.total_loss` |
| Table 6 ablation (12 variants) | `variants.VARIANTS` |
| Comparators (host, TI-Aug, FAS-Aug proxy) | `variants.build_augmentation` |
| Hosts ResNet-18, ViT-ConvPass, CLIP ViT | `models.build_model` |
| Protocols OCIM, MICY, limited, modern, WMCA unseen, OULU P3/P4 | `splits.make_split` |
| HTER at a source-fixed threshold, AUC, ACER (worst-case PAI) | `metrics` |
| Paired t-tests, Holm-Bonferroni, pooled CI and d_z | `stats.analyze`, `scripts/statistics.py` |
| Efficiency (Table 11) | `scripts/benchmark_efficiency.py` |
| Grad-CAM attention source (ablation) | `attention.GradCAMTrace` |
| Face detection, alignment, crops | `faces`, `scripts/prepare_faces.py` |

Use in another code base: [docs/PLUGIN.md](docs/PLUGIN.md).

## Fidelity notes

The numbers in the manuscript were produced by the authors' original experiment code. This
repository re-implements the method from its written description, so:

1. **Results have not yet been re-validated here.** Run the suites and compare the generated tables
   with the manuscript before citing numbers from this code.
2. **Comparators.** `fas_aug_proxy` is a class-agnostic stand-in for FAS-Aug, not the official
   implementation; the head-to-head claim needs the official code (see docs/REPRODUCING.md). The
   `clip_vit` host is image-only (CLIP ViT-B/16 + linear head); the full FLIP-MCL host needs its
   official code with APR-TMA plugged in (docs/PLUGIN.md). Numbers of other published methods are
   quoted from their papers and are not produced here.
3. **Unspecified details.** Table 2 does not fully pin down some construction choices. We chose:
   the CCM (rows from a Dirichlet draw, interpolated towards the identity with weight
   `U(0.1, 0.4)`, rows normalized), the filmic tone curve, the recapture scale `U(0.5, 0.9)`, the
   moire geometry (product of two sinusoids with random frequency and orientation), the surface
   operations of the "no internal ISP" ablation, and alpha_L = 0.20 / alpha_S = 0.50 for the
   class-dependent-magnitude ablation.
4. **SARE.** `losses.sare_penalty` implements risk equalization as the variance of group-wise mean
   losses over (class, augmented / clean). The FAS-Aug reference formulation may differ.
5. **VNG demosaicing** uses OpenCV and therefore runs on 8-bit data.
6. **Facial classes** of the parser used for `A(x)`: skin, brows, eyes, nose, mouth, lips (smoothed
   with a Gaussian).
7. **ViT-ConvPass** is a compact re-implementation (frozen ViT-B/16, bottleneck 8, two adapters per
   block), not the original ConvPass code. STDN trace maps for one ablation variant must come from
   the STDN authors' code.
8. **ACER** follows ISO/IEC 30107-3: APCER is the worst case over attack types when the manifest
   provides them, otherwise the pooled spoof acceptance rate.

## Repository layout

```
apr_tma/        pipeline, tma, augment, variants, losses, metrics, models, data, splits,
                stats, attention, faces, manifests
scripts/        prepare_faces, build_manifest, precompute_traces, run_suite, make_tables,
                statistics, benchmark_efficiency, make_qualitative_figure
configs/        suites.yaml (all experiment suites)
docs/           DATA.md, REPRODUCING.md, PLUGIN.md
tests/          unit tests for the augmentation, variants, protocols, statistics and data tools
train.py        one run (protocol, target, backbone, method / variant, seed)
```

## Citation

See `CITATION.cff`. Please update it with the journal reference once the article is published.

## License

MIT, see `LICENSE`.
