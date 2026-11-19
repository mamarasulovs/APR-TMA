# Reproducing the experiments

Prerequisites: [data preparation](DATA.md) (manifest + cached trace priors) and a GPU. All jobs use
the fixed seeds `{3407, 4519, 5633, 6749, 7861}` and the settings of the manuscript. Finished runs
are skipped, so every command can be repeated to resume.

```bash
M=data/manifest.csv; T=cache/traces
python scripts/run_suite.py all --manifest $M --trace-root $T --dry-run   # list the jobs
python scripts/run_suite.py ocim_vit --manifest $M --trace-root $T --amp   # one table
```

Use `--shard i/n` to spread a suite over several GPUs or machines, `--targets` to restrict folds
and `--seeds` for a subset of seeds.

## Manuscript item to command

| Item | Content | Suite / script |
|---|---|---|
| Table 3 | OCIM, ResNet-18 host, comparators | `run_suite.py ocim_resnet18` |
| Table 4 | OCIM, ViT-ConvPass host, head-to-head | `run_suite.py ocim_vit` |
| Table 5 | OCIM, vision-language host with APR-TMA plug-in | `run_suite.py ocim_clip_plugin` (image-only CLIP host; see PLUGIN.md for FLIP-MCL) |
| Table 6, Fig. 5 | component / leakage ablation | `run_suite.py ablation` |
| Table 7 | MICY | `run_suite.py micy` |
| Table 8 | limited-source (M & I) | `run_suite.py limited` |
| Table 9 | CelebA-Spoof <-> SiW-Mv2, WMCA unseen | `run_suite.py modern`, `run_suite.py wmca_unseen` |
| Table 10 | OULU-NPU protocols 3 and 4 | `run_suite.py oulu_p3`, `run_suite.py oulu_p4` |
| Table 11 | training cost, inference cost | `scripts/benchmark_efficiency.py` |
| Table 12, Fig. 7 | paired tests, Holm-Bonferroni, pooled CI | `scripts/statistics.py` |
| Fig. 6 | qualitative live / spoof example | `scripts/make_qualitative_figure.py` |

Then build all tables:

```bash
python scripts/make_tables.py --runs runs --out tables      # markdown, CSV and LaTeX per table
python scripts/statistics.py  --runs runs --out tables      # Table 12 + forest plot
python scripts/benchmark_efficiency.py --out tables/efficiency.json
```

## Notes on faithfulness

* `fas_aug_proxy` is a class-agnostic stand-in for FAS-Aug (colour / noise / blur / JPEG and
  recapture-style artifacts, with SARE), **not** the official implementation. For the head-to-head
  claim of Table 4 and the statistics of Table 12, run the official FAS-Aug code under the same
  backbone and protocol, save its `result.json` files in the same format, and point
  `scripts/statistics.py --baseline` at it.
* Numbers quoted from other papers in Tables 3-5, 7 and 8 (MADDG, SSDG, FLIP-MCL, ...) are taken
  from their publications and are not produced by this repository.
* The ablation variants are defined in `apr_tma/variants.py`; each removes or replaces exactly one
  element. Choices not fixed by the manuscript (for example alpha_L = 0.20 and alpha_S = 0.50 for the
  class-dependent magnitude) are documented there.
* Threshold and model selection use only the source-domain validation split (or the official `dev`
  subset for OULU-NPU). The target domain is never used before final scoring.
