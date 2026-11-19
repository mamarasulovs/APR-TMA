# Using APR-TMA as a plug-in in another code base

APR-TMA touches only the training images, so it can be inserted into any data pipeline (for example
the FLIP-MCL training code used for the plug-in experiment of the manuscript).

```python
import numpy as np
from apr_tma import APRTMA, APRTMAConfig

augment = APRTMA(APRTMAConfig())               # p_APR=0.50, lambda=0.70, alpha=0.35 (Table 2)

def transform(img_rgb_uint8, label, index, epoch, seed=3407):
    x = img_rgb_uint8.astype(np.float32) / 255.0              # 224x224 aligned face crop
    rng = np.random.default_rng([seed, epoch, index])        # reproducible per sample and epoch
    x_tilde, applied = augment(x, int(label), rng, trace=cached_trace_or_None)
    return x_tilde, applied                                   # feed x_tilde to the host's own
                                                              # normalisation and tensor conversion
```

* Apply it **before** the host's normalisation, to RGB float images in `[0, 1]`.
* Use the cached BiSeNet masks (`trace=`) so that the parser never runs in the data loader.
* Keep the host's own loss. For SARE-based hosts add `gamma * sare_penalty(...)` from
  `apr_tma.losses` with `gamma = 0.1`; the `augmented` flag returned above identifies the groups.
* Do not change the host's inference path: the deployed model stays identical to its backbone.
