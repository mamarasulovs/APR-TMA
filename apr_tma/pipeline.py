"""Randomized camera acquisition pipeline P_theta and reproduction layer R_xi.

Implements the stages listed in Table 2 of the manuscript:

    sRGB -> inverse tone map -> inverse gamma -> inverse CCM -> inverse white balance
         -> Bayer mosaic -> shot / read noise -> demosaic (MHC | VNG)
         -> white balance -> CCM -> tone map -> sharpen -> sRGB

All images are float32 arrays of shape (H, W, 3) with values in [0, 1].
The module is deterministic given a ``numpy.random.Generator``.

Parameters that Table 2 does not pin down (for example the exact CCM
construction and the moire geometry) are documented where they are defined.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict

import cv2
import numpy as np

BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
DEMOSAIC_OPS = ("MHC", "VNG")
TONE_CURVES = ("gamma", "filmic", "scurve")
RESIZE_KERNELS = ("bilinear", "bicubic", "lanczos")

_EPS = 1e-6
_CV_INTERP = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


# --------------------------------------------------------------------------- #
# Stage control (used by the ablation variants of Table 6)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageControl:
    """Switches that remove or replace exactly one element of the pipeline.

    * ``isp="surface"``        replace the internal ISP stages by surface-level operations.
    * ``symmetric_depth``      both classes use a single pass (no class asymmetry).
    * ``randomize=False``      asymmetric structure, but fixed (canonical) parameters.
    * ``drop_*``               fix tone / white balance / gamma, remove shot and read noise,
                               fix the demosaicing operator and CFA pattern, or remove sharpening.
    """

    isp: str = "internal"
    symmetric_depth: bool = False
    randomize: bool = True
    drop_tone_wb: bool = False
    drop_noise: bool = False
    drop_demosaic_rand: bool = False
    drop_sharpen: bool = False


DEFAULT_CONTROL = StageControl()


# --------------------------------------------------------------------------- #
# Parameter container and samplers
# --------------------------------------------------------------------------- #
def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


@dataclass
class PipelineParams:
    """Parameters theta of one acquisition pass (distributions: Table 2)."""

    inv_gamma: float = 2.2
    inv_tone_strength: float = 0.8
    red_gain: float = 2.0
    blue_gain: float = 1.6
    bayer: str = "RGGB"
    shot_scale: float = 1e-3
    read_std: float = 1e-3
    demosaic: str = "MHC"
    ccm_mix: float = 0.2
    ccm_dirichlet: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    tone_curve: str = "gamma"
    sharp_radius: float = 0.8
    sharp_amount: float = 0.5
    noise_on: bool = True

    # names of the continuous / categorical groups, used by the coupling rule
    CONTINUOUS = ("inv_gamma", "inv_tone_strength", "red_gain", "blue_gain",
                  "shot_scale", "read_std", "ccm_mix", "sharp_radius", "sharp_amount")
    CATEGORICAL = ("bayer", "demosaic")

    @staticmethod
    def sample(rng: np.random.Generator, ctl: StageControl = DEFAULT_CONTROL) -> "PipelineParams":
        if not ctl.randomize:
            return apply_stage_drops(PipelineParams(), ctl)
        return apply_stage_drops(PipelineParams._draw(rng), ctl)

    @staticmethod
    def _draw(rng: np.random.Generator) -> "PipelineParams":
        return PipelineParams(
            inv_gamma=rng.uniform(1.8, 2.6),
            inv_tone_strength=rng.uniform(0.6, 1.0),
            red_gain=_log_uniform(rng, 1.6, 2.4),
            blue_gain=_log_uniform(rng, 1.3, 2.1),
            bayer=str(rng.choice(BAYER_PATTERNS)),
            shot_scale=_log_uniform(rng, 1e-4, 1.2e-2),
            read_std=_log_uniform(rng, 1e-4, 4e-3),
            demosaic=str(rng.choice(DEMOSAIC_OPS)),
            # Table 2 only states "Dirichlet mixture, normalize rows". We draw each CCM row
            # from Dirichlet(1,1,1) and interpolate towards the identity with weight ccm_mix.
            ccm_mix=rng.uniform(0.1, 0.4),
            ccm_dirichlet=rng.dirichlet(np.ones(3), size=3).astype(np.float32),
            tone_curve=str(rng.choice(TONE_CURVES)),
            sharp_radius=rng.uniform(0.3, 1.5),
            sharp_amount=rng.uniform(0.0, 1.2),
        )

    def ccm(self) -> np.ndarray:
        m = (1.0 - self.ccm_mix) * np.eye(3, dtype=np.float32) + self.ccm_mix * self.ccm_dirichlet
        return m / m.sum(axis=1, keepdims=True)  # rows sum to one: white stays white


def apply_stage_drops(p: PipelineParams, ctl: StageControl) -> PipelineParams:
    """Fix or remove the stages switched off in ``ctl`` (see :class:`StageControl`)."""
    base = PipelineParams()
    if ctl.drop_tone_wb:
        p = replace(p, inv_gamma=base.inv_gamma, inv_tone_strength=base.inv_tone_strength,
                    red_gain=base.red_gain, blue_gain=base.blue_gain, ccm_mix=0.0,
                    ccm_dirichlet=base.ccm_dirichlet, tone_curve="gamma")
    if ctl.drop_noise:
        p = replace(p, noise_on=False)
    if ctl.drop_demosaic_rand:
        p = replace(p, bayer="RGGB", demosaic="MHC")
    if ctl.drop_sharpen:
        p = replace(p, sharp_amount=0.0)
    return p


def couple_params(first: PipelineParams, rng: np.random.Generator,
                  rho: float | None = None, q_c: float = 0.5,
                  ctl: StageControl = DEFAULT_CONTROL) -> PipelineParams:
    """Second-pass parameters, Eq. (5).

    Continuous parameters: theta_2 = rho * theta_1 + (1 - rho) * theta_tilde, rho ~ Beta(2, 2).
    Categorical parameters are re-sampled with probability ``q_c`` and otherwise inherited.
    """
    if rho is None:
        rho = float(rng.beta(2.0, 2.0))
    fresh = PipelineParams.sample(rng, ctl)
    updates: Dict[str, object] = {}
    for name in PipelineParams.CONTINUOUS:
        updates[name] = rho * getattr(first, name) + (1.0 - rho) * getattr(fresh, name)
    updates["ccm_dirichlet"] = rho * first.ccm_dirichlet + (1.0 - rho) * fresh.ccm_dirichlet
    for name in PipelineParams.CATEGORICAL:
        updates[name] = getattr(fresh, name) if rng.random() < q_c else getattr(first, name)
    updates["tone_curve"] = fresh.tone_curve  # "independent per pass" (Table 2)
    return apply_stage_drops(replace(first, **updates), ctl)


@dataclass
class ReproductionParams:
    """Parameters xi of the reproduction layer R_xi = A_phi o D_psi (Table 2)."""

    resize_kernel: str = "bicubic"
    blur_sigma: float = 0.6
    moire_strength: float = 0.15
    scale: float = 0.7
    moire_freq: tuple = (0.12, 0.17)
    moire_angle: float = 0.4

    @staticmethod
    def sample(rng: np.random.Generator, ctl: StageControl = DEFAULT_CONTROL) -> "ReproductionParams":
        if not ctl.randomize:
            return ReproductionParams()
        return ReproductionParams(
            resize_kernel=str(rng.choice(RESIZE_KERNELS)),
            blur_sigma=rng.uniform(0.0, 1.2),
            moire_strength=rng.uniform(0.0, 0.35),
            # The manuscript gives no ranges for the recapture geometry; these are our choices.
            scale=rng.uniform(0.5, 0.9),
            moire_freq=(rng.uniform(0.08, 0.25), rng.uniform(0.08, 0.25)),
            moire_angle=rng.uniform(0.0, np.pi),
        )


# --------------------------------------------------------------------------- #
# Tone mapping
# --------------------------------------------------------------------------- #
def _smoothstep(x: np.ndarray) -> np.ndarray:
    return 3.0 * x ** 2 - 2.0 * x ** 3


def _inverse_smoothstep(y: np.ndarray) -> np.ndarray:
    """Inverse of the smoothstep curve (Brooks et al., 2019)."""
    y = np.clip(y, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * y) / 3.0)


def _tone_map(x: np.ndarray, curve: str, gamma: float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    if curve == "gamma":
        return x ** (1.0 / gamma)
    if curve == "filmic":
        y = x * (1.0 + x / 4.0) / (1.0 + x)  # Reinhard-style shoulder, maps 1 -> 1.25 / 2
        y = y / (1.0 * (1.0 + 1.0 / 4.0) / 2.0)
        return np.clip(y, 0.0, 1.0) ** (1.0 / gamma)
    if curve == "scurve":
        return _smoothstep(x ** (1.0 / gamma))
    raise ValueError(curve)


# --------------------------------------------------------------------------- #
# CFA: mosaic and demosaic
# --------------------------------------------------------------------------- #
def _cfa_masks(shape, pattern: str):
    h, w = shape
    r = np.zeros((h, w), bool)
    g = np.zeros((h, w), bool)
    b = np.zeros((h, w), bool)
    layout = {c: (i // 2, i % 2) for i, c in enumerate(pattern)}  # position in the 2x2 cell
    # A letter may appear twice (the two greens), so iterate over cell positions instead.
    for idx, ch in enumerate(pattern):
        dy, dx = divmod(idx, 2)
        tgt = {"R": r, "G": g, "B": b}[ch]
        tgt[dy::2, dx::2] = True
    del layout
    return r, g, b


def mosaic(img: np.ndarray, pattern: str) -> np.ndarray:
    r, g, b = _cfa_masks(img.shape[:2], pattern)
    cfa = np.zeros(img.shape[:2], np.float32)
    cfa[r] = img[..., 0][r]
    cfa[g] = img[..., 1][g]
    cfa[b] = img[..., 2][b]
    return cfa


# Malvar-He-Cutler (2004) gradient-corrected linear interpolation kernels.
_GR_GB = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                   [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32) / 8.0
_RG_RB_BG_BR = np.array([[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                         [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], np.float32) / 8.0
_RG_BR_BG_RB = _RG_RB_BG_BR.T.copy()
_RB_BR_BR_RB = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                         [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32) / 8.0


def _conv(cfa: np.ndarray, k: np.ndarray) -> np.ndarray:
    return cv2.filter2D(cfa, -1, k, borderType=cv2.BORDER_REFLECT_101)


def demosaic_mhc(cfa: np.ndarray, pattern: str) -> np.ndarray:
    """Malvar-He-Cutler demosaicing for any of the four Bayer layouts."""
    rm, gm, bm = _cfa_masks(cfa.shape, pattern)
    R, G, B = cfa * rm, cfa * gm, cfa * bm
    G = np.where(rm | bm, _conv(cfa, _GR_GB), G)
    rbg_rbbr = _conv(cfa, _RG_RB_BG_BR)
    rbg_brrb = _conv(cfa, _RG_BR_BG_RB)
    rbgr_bbrr = _conv(cfa, _RB_BR_BR_RB)
    r_row = np.any(rm, axis=1)[:, None] * np.ones_like(rm)
    r_col = np.any(rm, axis=0)[None, :] * np.ones_like(rm)
    b_row = np.any(bm, axis=1)[:, None] * np.ones_like(bm)
    b_col = np.any(bm, axis=0)[None, :] * np.ones_like(bm)
    R = np.where(r_row & b_col, rbg_rbbr, R)
    R = np.where(b_row & r_col, rbg_brrb, R)
    B = np.where(b_row & r_col, rbg_rbbr, B)
    B = np.where(r_row & b_col, rbg_brrb, B)
    R = np.where(b_row & b_col, rbgr_bbrr, R)
    B = np.where(r_row & r_col, rbgr_bbrr, B)
    return np.stack([R, G, B], axis=-1)


def _vng_code() -> int:
    """OpenCV's Bayer constant naming is confusing; pick the code that decodes RGGB correctly."""
    global _VNG_CODE
    if _VNG_CODE is not None:
        return _VNG_CODE
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32) / 31.0
    ref = np.stack([xx, yy, 0.5 * (xx + yy)], -1)
    cfa8 = np.clip(mosaic(ref, "RGGB") * 255.0, 0, 255).astype(np.uint8)
    best, best_err = None, 1e9
    for name in ("COLOR_BayerBG2RGB_VNG", "COLOR_BayerRG2RGB_VNG",
                 "COLOR_BayerGB2RGB_VNG", "COLOR_BayerGR2RGB_VNG"):
        code = getattr(cv2, name)
        out = cv2.cvtColor(cfa8, code).astype(np.float32) / 255.0
        err = float(np.abs(out[4:-4, 4:-4] - ref[4:-4, 4:-4]).mean())
        if err < best_err:
            best, best_err = code, err
    _VNG_CODE = best
    return best


_VNG_CODE = None
_PAD = {"RGGB": (0, 0), "GRBG": (0, 1), "GBRG": (1, 0), "BGGR": (1, 1)}  # (rows, cols)


def demosaic_vng(cfa: np.ndarray, pattern: str) -> np.ndarray:
    """Variable-number-of-gradients demosaicing via OpenCV (8-bit internally).

    Any layout is reduced to RGGB by reflect-padding one row and/or column, which preserves the
    CFA parity, decoding as RGGB, and cropping the padding afterwards.
    """
    py, px = _PAD[pattern]
    padded = np.pad(cfa, ((py, 0), (px, 0)), mode="reflect")
    cfa8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cfa8, _vng_code()).astype(np.float32) / 255.0
    return out[py:, px:]


def demosaic(cfa: np.ndarray, pattern: str, op: str) -> np.ndarray:
    if op == "MHC":
        return demosaic_mhc(cfa, pattern)
    if op == "VNG":
        return demosaic_vng(cfa, pattern)
    raise ValueError(op)


# --------------------------------------------------------------------------- #
# One acquisition pass
# --------------------------------------------------------------------------- #
def acquisition_pass(img: np.ndarray, p: PipelineParams, rng: np.random.Generator) -> np.ndarray:
    """Apply P_theta to an sRGB image in [0, 1]. Returns an sRGB image in [0, 1]."""
    x = np.clip(img.astype(np.float32), 0.0, 1.0)

    # --- pseudo-RAW inversion (approximate, monotone) ---
    inv = _inverse_smoothstep(x)
    x = p.inv_tone_strength * inv + (1.0 - p.inv_tone_strength) * x
    x = np.clip(x, 0.0, 1.0) ** p.inv_gamma
    ccm = p.ccm()
    x = np.clip(x @ np.linalg.inv(ccm).T, 0.0, 1.0)
    gains = np.array([p.red_gain, 1.0, p.blue_gain], np.float32)
    x = np.clip(x / gains * gains.min(), 0.0, 1.0)  # inverse WB (keeps highlights unclipped)

    # --- CFA sampling and sensor noise in the Bayer domain ---
    cfa = mosaic(x, p.bayer)
    if p.noise_on:
        shot = rng.poisson(np.clip(cfa, 0, 1) / p.shot_scale).astype(np.float32) * p.shot_scale
        cfa = shot + rng.normal(0.0, p.read_std, cfa.shape).astype(np.float32)
        cfa = np.clip(cfa, 0.0, 1.0)

    # --- processing back to sRGB ---
    x = np.clip(demosaic(cfa, p.bayer, p.demosaic), 0.0, 1.0)
    x = np.clip(x * gains / gains.min(), 0.0, 1.0)  # white balance
    x = np.clip(x @ ccm.T, 0.0, 1.0)
    x = _tone_map(x, p.tone_curve, p.inv_gamma)

    # --- sharpening (unsharp mask) ---
    if p.sharp_amount > 1e-3:
        blur = cv2.GaussianBlur(x, (0, 0), sigmaX=max(p.sharp_radius, 0.1))
        x = x + p.sharp_amount * (x - blur)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reproduction layer: print / screen rendering + recapture geometry
# --------------------------------------------------------------------------- #
def reproduce(img: np.ndarray, xi: ReproductionParams) -> np.ndarray:
    """R_xi: resize kernel, medium blur and medium-conditioned moire / halftone term."""
    h, w = img.shape[:2]
    interp = _CV_INTERP[xi.resize_kernel]
    small = cv2.resize(img, (max(8, int(w * xi.scale)), max(8, int(h * xi.scale))),
                       interpolation=cv2.INTER_AREA)
    x = cv2.resize(small, (w, h), interpolation=interp)
    if xi.blur_sigma > 1e-3:
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=xi.blur_sigma)
    if xi.moire_strength > 1e-3:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        c, s = np.cos(xi.moire_angle), np.sin(xi.moire_angle)
        u = xx * c + yy * s
        v = -xx * s + yy * c
        grid = np.sin(2 * np.pi * xi.moire_freq[0] * u) * np.sin(2 * np.pi * xi.moire_freq[1] * v)
        x = x + 0.5 * xi.moire_strength * grid[..., None]
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Surface-level operations (replacement of the internal ISP in the ablation; also TI-Aug)
# --------------------------------------------------------------------------- #
def surface_pass(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generic image-space distortions with no camera model: colour, noise, blur, JPEG."""
    x = np.clip(img.astype(np.float32), 0.0, 1.0)
    x = np.clip((x - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.15, 0.15), 0.0, 1.0)
    x = x ** rng.uniform(0.7, 1.4)
    x = x * rng.uniform(0.9, 1.1, size=3).astype(np.float32)          # per-channel gain
    x = x + rng.normal(0.0, rng.uniform(0.0, 0.03), x.shape).astype(np.float32)
    sigma = rng.uniform(0.0, 1.2)
    if sigma > 1e-3:
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma)
    q = int(rng.integers(40, 96))
    ok, buf = cv2.imencode(".jpg", (np.clip(x, 0, 1) * 255).astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        x = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Class-conditional composition, Eq. (3) and (4)
# --------------------------------------------------------------------------- #
def acquisition_depth(y: int, ctl: StageControl = DEFAULT_CONTROL) -> int:
    """K_y = 1 + y (Eq. 3); a symmetric composition uses K = 1 for both classes."""
    return 1 if ctl.symmetric_depth else 1 + int(y)


def _one_pass(img, theta, rng, ctl):
    if ctl.isp == "surface":
        return surface_pass(img, rng)
    return acquisition_pass(img, theta, rng)


def apply_pipeline(img: np.ndarray, y: int, rng: np.random.Generator, q_c: float = 0.5,
                   ctl: StageControl = DEFAULT_CONTROL) -> np.ndarray:
    """P(x) of Eq. (4): one pass for live (y = 0), two coupled passes plus reproduction for spoof."""
    theta1 = PipelineParams.sample(rng, ctl)
    out = _one_pass(img, theta1, rng, ctl)
    if acquisition_depth(y, ctl) == 2:
        xi = ReproductionParams.sample(rng, ctl)
        theta2 = couple_params(theta1, rng, q_c=q_c, ctl=ctl)
        out = _one_pass(reproduce(out, xi), theta2, rng, ctl)
    return out
