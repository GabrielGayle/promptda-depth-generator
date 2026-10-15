"""Grey PNG for Strata (laser-cut plate reliefs) from a metric depth run.

Strata reads any image as a 0..1 brightness grid where dark = deep / proud
(Raise lifts dark, Front "Layered cut-outs" cuts on dark, "Ignore lighter
than" leaves white uncut). So the plate here is: near subject = dark,
background = white, 8-bit, small (Strata downsamples to 256 px anyway).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

STRATA_MODES = ("relief", "depth")
DEFAULT_STRATA_MODE = "relief"
DEFAULT_STRATA_MAX_SIDE = 512
DEFAULT_STRATA_GC_ALPHA = 10.0
DEFAULT_STRATA_GAMMA = 1.0


def render_strata_plate(
    depth_m: np.ndarray,
    *,
    cliff: Optional[float] = None,
    mode: str = DEFAULT_STRATA_MODE,
    matte: bool = True,
    gc_alpha: float = DEFAULT_STRATA_GC_ALPHA,
    gamma: float = DEFAULT_STRATA_GAMMA,
    max_side: int = DEFAULT_STRATA_MAX_SIDE,
) -> np.ndarray:
    """HxW uint8 grey: 0 = nearest / highest, 255 = far or matted-out background.

    mode "depth": inverse depth, percentile-windowed on the subject.
    mode "relief": the same, then Weyrich bas-relief compression so fine
    detail survives being squashed into a few millimetres of plate.
    """
    from depth_core.pipeline import _prepare_inv_smooth

    depth = np.asarray(depth_m, dtype=np.float64)
    valid, keep = _subject_keep(depth, matte=matte, cliff=cliff)
    _, inv = _prepare_inv_smooth(depth)

    region = keep > 0.5
    lo, hi = np.percentile(inv[region], [2.0, 98.0])
    h = np.clip((inv - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    if gamma and abs(gamma - 1.0) > 1e-6:
        h = h ** float(gamma)

    if mode == "relief":
        from depth_core.relief import render_bas_relief

        src = h * keep
        rel = render_bas_relief(
            (src * 255.0).astype(np.uint8), gc_alpha=gc_alpha, max_side=512
        )[..., 0].astype(np.float64) / 255.0
        # The Poisson solve's sign can come back flipped (bright input -> dark
        # relief); orient it so near stays high whatever relief.py returns.
        if np.corrcoef(rel.ravel(), src.ravel())[0, 1] < 0:
            rel = 1.0 - rel
        # Re-zero the background so it stays white (uncut) in Strata.
        rel = rel - float(np.percentile(rel[~region], 50)) if (~region).any() else rel
        h = np.clip(rel / max(float(rel[region].max()), 1e-9), 0.0, 1.0)
    elif mode != "depth":
        raise ValueError(f"Unknown Strata mode {mode!r}; use one of {STRATA_MODES}")

    h = h * keep
    grey = 1.0 - h  # near = dark, background = white

    rows, cols = grey.shape
    side = int(max(64, max_side))
    if max(rows, cols) > side:
        s = side / float(max(rows, cols))
        grey = cv2.resize(
            grey, (max(2, round(cols * s)), max(2, round(rows * s))),
            interpolation=cv2.INTER_AREA,
        )
    return np.clip(np.round(grey * 255.0), 0, 255).astype(np.uint8)


def _subject_keep(
    depth: np.ndarray, *, matte: bool, cliff: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """(valid mask, 0..1 subject matte). Without matte the whole valid scene is kept."""
    from depth_core.pipeline import (
        _scenery_depth_window,
        _subject_hi_pct,
        _subject_matte,
        depth_valid_mask,
    )

    valid = depth_valid_mask(depth)
    if not valid.any():
        raise ValueError("Depth has no valid pixels")
    keep = valid.astype(np.float32)
    if matte:
        c = float(cliff) if cliff is not None else float(_subject_hi_pct(depth))
        near, _far, cut = _scenery_depth_window(depth, c)
        keep = np.clip(_subject_matte(depth, cut, near), 0.0, 1.0).astype(np.float32)
        keep *= valid
        if float(keep.max()) < 0.5:
            keep = valid.astype(np.float32)  # no clear subject: keep whole scene
    return valid, keep


def strata_depth_payload(
    depth_m: np.ndarray,
    *,
    matte: bool = True,
    cliff: Optional[float] = None,
    max_side: int = 512,
) -> dict:
    """The metric depth itself, for Strata to window and shape in the browser.

    data: base64 little-endian uint16, row-major. 0 = no depth (sky / invalid);
    1..65535 map linearly onto min_m..max_m metres. mask: base64 uint8 subject
    matte (255 = keep) or None. near_m / far_m: the window the grey plate uses
    (2nd / 98th percentile of inverse depth over the subject), as a starting point.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    valid, keep = _subject_keep(depth, matte=matte, cliff=cliff)

    region = keep > 0.5
    inv = 1.0 / np.clip(depth[region], 1e-4, None)
    lo, hi = np.percentile(inv, [2.0, 98.0])
    near_m, far_m = 1.0 / max(float(hi), 1e-9), 1.0 / max(float(lo), 1e-9)

    vals = depth[valid]
    d_min = float(vals.min())
    d_max = max(float(np.percentile(vals, 99.5)), far_m * 1.5, d_min + 1e-3)

    rows, cols = depth.shape
    side = int(max(64, max_side))
    if max(rows, cols) > side:
        s = side / float(max(rows, cols))
        size = (max(2, round(cols * s)), max(2, round(rows * s)))
        filled = np.where(valid, np.clip(depth, d_min, d_max), 0.0).astype(np.float32)
        w = cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(filled, size, interpolation=cv2.INTER_AREA) / np.maximum(w, 1e-6)
        valid = w > 0.5
        keep = cv2.resize(keep, size, interpolation=cv2.INTER_AREA)

    q = np.zeros(depth.shape, dtype="<u2")
    t = (np.clip(depth, d_min, d_max) - d_min) / (d_max - d_min)
    q[valid] = (1 + np.round(t[valid] * 65534)).astype("<u2")
    mask = None
    if matte:
        mask = base64.b64encode(np.clip(np.round(keep * 255), 0, 255).astype(np.uint8).tobytes()).decode()
    return {
        "w": int(q.shape[1]),
        "h": int(q.shape[0]),
        "min_m": d_min,
        "max_m": d_max,
        "near_m": near_m,
        "far_m": far_m,
        "data": base64.b64encode(q.tobytes()).decode(),
        "mask": mask,
    }


def export_strata_png(
    out_dir: Path | str,
    run_id: str,
    depth_m: np.ndarray,
    **kwargs,
) -> Path:
    import imageio

    plate = render_strata_plate(depth_m, **kwargs)
    path = Path(out_dir) / f"{run_id}_strata_{kwargs.get('mode', DEFAULT_STRATA_MODE)}.png"
    imageio.imwrite(path, plate)
    return path
