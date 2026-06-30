"""Topology-conserving 4D-flow velocity filter.

Background
----------
The raw 4D-flow export is a uniform ImageData grid (137 x 227 x 27 cells) with a
cell-centred ``velocity`` field, plus a binary ``mask`` that marks the segmented
lumen.  Across the cardiac cycle there are 21 time frames.  Inside the mask every
voxel carries a velocity at every frame; outside the mask the velocity is zero.

The legacy filter (which we never received) only looked at the *surface* and only
asked "is this vector temporally inconsistent with its neighbourhood?".  This
script does something more deliberate, following the two-outcome strategy in the
README:

  1. CHRONIC voxels -- voxels that are *spatially* wrong in many frames.  These are
     almost always on the mask boundary (segmentation error / chronic partial
     voluming).  They are **excluded** from the lumen entirely.

  2. TRANSIENT voxel-frames -- voxels (boundary OR core) that look fine most of
     the cycle but spike for a handful of frames.  Those individual frames are
     **replaced** by a spatiotemporal average of their good neighbours, which
     conserves topology (the voxel stays part of the lumen).

Why spatial vs. temporal residuals are used differently
-------------------------------------------------------
A voxel that is *consistently* wrong every frame is temporally smooth, so a
temporal test misses it.  Hence the **spatial** residual is the chronic detector,
while the **temporal** residual catches single-frame spikes in otherwise-good
voxels.

Detection is robust: thresholds are set from the median + k * MAD of the residual
distribution, and the neighbour estimate is recomputed once with gross outliers
removed so a bad voxel does not poison its own neighbourhood statistic.

Run ``python new_4dflow_filter.py`` to process the dataset and write filtered
frames + an updated mask + a diagnostics report into ``./output``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import ndimage


@dataclass
class FilterConfig:
    """Tunable knobs.  Defaults were chosen from the residual distributions of
    this dataset (see the README analysis), but everything here is data-driven
    via MAD, so the same defaults transfer to other acquisitions.
    """

    # Robust threshold strength: flag a residual above median + k * MAD.
    spatial_k: float = 5.0
    temporal_k: float = 5.0
    # A voxel is "chronic" (excluded) if it is a spatial outlier in at least this
    # fraction of frames.
    chronic_fraction: float = 0.40
    # 3x3x3 spatial neighbourhood (26-connected) used for both detection and
    # inpainting.  Temporal half-window for inpainting (+/- this many frames).
    temporal_halfwidth: int = 1
    # Max iterations when inpainting (lets holes whose neighbours are also bad
    # fill in over successive passes).
    inpaint_iters: int = 8
    # Treat the cardiac cycle as periodic (frame 0 follows frame T-1) for the
    # temporal residual and inpainting.  4D-flow cines are gated, so this holds.
    periodic_time: bool = True
    # Also fill enclosed cavities that already exist in the input segmentation
    # (holes fully surrounded by lumen).  These violate downstream-solver
    # topology too, so by default we add them to the lumen and inpaint their
    # velocity.  Set False to leave the input segmentation's holes untouched.
    fill_preexisting_cavities: bool = True
    # Discard everything except the largest face-connected lumen component.
    # Removes partial-volume "speckle" voxels that touch the body only through an
    # edge/corner (disconnected under the solver's 6-connectivity).
    keep_largest_component: bool = True


# 3x3x3 footprint with the centre removed (26 neighbours).
_FOOT = np.ones((3, 3, 3), dtype=np.float32)
_FOOT[1, 1, 1] = 0.0

# Face connectivity (6-neighbour).  A downstream solver connects cells through
# shared faces, so "no enclosed cavity" means the background must be
# face-connected to the exterior.
_FACE = ndimage.generate_binary_structure(3, 1)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_stack(data_dir: Path, n: int, venc: float = 2.0):
    """Load every frame into a dense ``(T, nz, ny, nx, 3)`` array plus the mask.

    Voxels outside the lumen are set to NaN so they are ignored by every
    neighbour statistic.  Returns ``(velocity, mask, template)`` where
    ``template`` is the first ImageData object (geometry to clone on write).
    """
    # FIXME: change hardcoded strings
    template = pv.read(data_dir / f"UM8_velocity_t{0:03d}.vti")
    inner_prod = np.einsum("ij,ij->i", template.cell_data["velocity"], template.cell_data["velocity"])
    nx, ny, nz = np.array(template.dimensions) - 1

    exterior_mask = (inner_prod > 1e-8) | (inner_prod < -1e-8)
    print(f"exterior_mask size: {exterior_mask.sum()}")
    out_of_bounds_mask = np.any(np.abs(template.cell_data["velocity"]) < venc, axis=1)
    print(f"out_of_bounds_mask size: {out_of_bounds_mask.sum()}")
    mask = (exterior_mask & out_of_bounds_mask).reshape(nz, ny, nx)
    print(f"total mask size: {mask.sum()}")
    # NOTE: add things about being below 2 per component?
    # mask = (pv.read(data_dir / "UM8_paraview_viz_mask.vti").cell_data["mask"] > 0).reshape(nz, ny, nx)

    vel = np.full((n, nz, ny, nx, 3), np.nan, dtype=np.float32)
    for i in range(n):
        frame = pv.read(data_dir / f"UM8_velocity_t{i:03d}.vti")
        vel[i] = frame.cell_data["velocity"].reshape(nz, ny, nx, 3)
    vel[:, ~mask] = np.nan
    return vel, mask, template


# ---------------------------------------------------------------------------
# Neighbour statistics
# ---------------------------------------------------------------------------
def _neighbour_mean(vec_field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean of the 26 spatial neighbours for a ``(nz, ny, nx, 3)`` field,
    counting only ``valid`` voxels.  Returns NaN where no valid neighbour.
    """
    out = np.full_like(vec_field, np.nan)
    counts = ndimage.convolve(valid.astype(np.float32), _FOOT, mode="constant")
    safe = np.maximum(counts, 1.0)
    for a in range(3):
        f = np.where(valid, vec_field[..., a], 0.0).astype(np.float32)
        s = ndimage.convolve(f, _FOOT, mode="constant")
        out[..., a] = np.where(counts > 0, s / safe, np.nan)
    return out


def _topology_safe_exclusion(chronic, mask):
    """Split chronic voxels into those we may *exclude* vs. those we must *keep*.

    A voxel can be excluded only if removing it does not punch a hole into the
    interior of the lumen.  Concretely: build ``background | chronic`` and find
    the component(s) face-connected to the domain exterior.  A chronic voxel in
    that exterior component is reachable from outside through other excluded /
    background voxels, so excluding it leaves no enclosed cavity -- this exactly
    encodes the rule "a one-layer-deep voxel is fine as long as the layer above
    it is also excluded".

    Any chronic voxel that is *not* exterior-connected would become an enclosed
    cavity (a topology violation for the solver), so it is kept in the lumen and
    flagged for inpainting instead.

    Returns ``(exclude, reclaim)`` boolean grids; ``exclude | reclaim ==
    chronic`` and they are disjoint.  Keeping the enclosed voxels can never
    block an exterior-connected voxel (an enclosed voxel is by definition not on
    any exterior path), so a single pass is sufficient.
    """
    passable = (~mask) | chronic
    lab, _ = ndimage.label(passable, structure=_FACE)
    border = np.concatenate(
        [
            lab[0].ravel(),
            lab[-1].ravel(),
            lab[:, 0].ravel(),
            lab[:, -1].ravel(),
            lab[:, :, 0].ravel(),
            lab[:, :, -1].ravel(),
        ],
    )
    exterior_labels = set(np.unique(border)) - {0}
    exterior = np.isin(lab, list(exterior_labels))
    exclude = chronic & exterior
    reclaim = chronic & ~exterior
    return exclude, reclaim


def _largest_component(lumen):
    """Boolean grid of the largest face-connected component of ``lumen``."""
    lab, n = ndimage.label(lumen, structure=_FACE)
    if n <= 1:
        return lumen.copy()
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # ignore background
    return lab == sizes.argmax()


def _enclosed_cavities(lumen):
    """Boolean grid of background voxels that are not face-connected to the
    exterior, i.e. holes fully surrounded by lumen.
    """
    lab, _ = ndimage.label(~lumen, structure=_FACE)
    border = np.concatenate(
        [
            lab[0].ravel(),
            lab[-1].ravel(),
            lab[:, 0].ravel(),
            lab[:, -1].ravel(),
            lab[:, :, 0].ravel(),
            lab[:, :, -1].ravel(),
        ],
    )
    exterior_labels = set(np.unique(border)) - {0}
    return (~lumen) & ~np.isin(lab, list(exterior_labels))


def _robust_threshold(values: np.ndarray, k: float) -> float:
    """Median + k * (1.4826 * MAD) over the finite entries."""
    v = values[np.isfinite(values)]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return float(med + k * mad)


def spatial_residual(vel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per voxel-frame magnitude of the deviation from the spatial-neighbour
    mean, ``(T, nz, ny, nx)``.

    Two passes: the first uses every masked neighbour, the second drops the
    grossly deviating voxels so a bad voxel cannot inflate its own neighbours'
    mean and hide itself.
    """
    T = vel.shape[0]
    res = np.full(vel.shape[:4], np.nan, dtype=np.float32)
    for t in range(T):
        nm = _neighbour_mean(vel[t], mask)
        res[t] = np.linalg.norm(vel[t] - nm, axis=-1)

    gross = res > _robust_threshold(res, k=6.0)
    res2 = np.full_like(res, np.nan)
    for t in range(T):
        good = mask & ~gross[t]
        nm = _neighbour_mean(vel[t], good)
        # Fall back to the first-pass mean where decontamination left no neighbour.
        fallback = _neighbour_mean(vel[t], mask)
        nm = np.where(np.isfinite(nm), nm, fallback)
        res2[t] = np.linalg.norm(vel[t] - nm, axis=-1)
    return res2


def temporal_residual(vel: np.ndarray, periodic: bool) -> np.ndarray:
    """Per voxel-frame deviation from the mean of the previous and next frame."""
    if periodic:
        prev, nxt = np.roll(vel, 1, axis=0), np.roll(vel, -1, axis=0)
    else:
        prev = np.concatenate([vel[:1], vel[:-1]], axis=0)
        nxt = np.concatenate([vel[1:], vel[-1:]], axis=0)
    return np.linalg.norm(vel - 0.5 * (prev + nxt), axis=-1)


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------
def inpaint(vel, keep_mask, replace_mask, cfg: FilterConfig):
    """Fill ``replace_mask`` voxel-frames with a spatiotemporal weighted average
    of *good* neighbours (``keep_mask & ~replace_mask``).

    Iterated so that a bad voxel whose neighbours are also bad still fills once
    its neighbours have been filled.  Spatial neighbours (26-connected, weight 1)
    and the +/- temporal_halfwidth temporal neighbours (weight 1) both vote.
    """
    out = vel.copy()
    out[replace_mask] = np.nan  # forget the bad values before we average
    good = keep_mask & ~replace_mask
    todo = replace_mask.copy()
    T = vel.shape[0]
    hw = cfg.temporal_halfwidth

    for _ in range(cfg.inpaint_iters):
        if not todo.any():
            break
        valid = good | (keep_mask & np.isfinite(out[..., 0]))
        num = np.zeros_like(out)
        den = np.zeros(out.shape[:4], dtype=np.float32)

        # Spatial votes, frame by frame.
        for t in range(T):
            nm = _neighbour_mean(out[t], valid[t])
            cnt = ndimage.convolve(
                valid[t].astype(np.float32),
                _FOOT,
                mode="constant",
            )
            contributes = np.isfinite(nm[..., 0])
            for a in range(3):
                num[t, ..., a] += np.where(contributes, nm[..., a] * cnt, 0.0)
            den[t] += np.where(contributes, cnt, 0.0)

        # Temporal votes.
        for dt in range(1, hw + 1):
            for shift in (dt, -dt):
                rolled_v = np.roll(out, shift, axis=0)
                rolled_ok = np.roll(valid, shift, axis=0)
                if not cfg.periodic_time:
                    edge = slice(0, shift) if shift > 0 else slice(shift, None)
                    rolled_ok[edge] = False
                ok = rolled_ok & np.isfinite(rolled_v[..., 0])
                for a in range(3):
                    num[..., a] += np.where(ok, rolled_v[..., a], 0.0)
                den += ok.astype(np.float32)

        fillable = todo & (den > 0)
        with np.errstate(invalid="ignore"):
            filled = num / np.maximum(den[..., None], 1.0)
        for a in range(3):
            out[..., a] = np.where(fillable, filled[..., a], out[..., a])
        todo &= ~fillable

    # Any voxel-frame we still could not fill: drop back to zero (isolated).
    out[todo] = 0.0
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
@dataclass
class FilterResult:
    velocity: np.ndarray  # cleaned (T, nz, ny, nx, 3)
    mask: np.ndarray  # updated lumen mask (nz, ny, nx) after chronic removal
    chronic: np.ndarray  # (nz, ny, nx) bool, voxels excluded
    replaced: np.ndarray  # (T, nz, ny, nx) bool, voxel-frames inpainted
    stats: dict = field(default_factory=dict)


def filter_4dflow(vel, mask, cfg: FilterConfig = FilterConfig()) -> FilterResult:
    T = vel.shape[0]
    sres = spatial_residual(vel, mask)
    tres = temporal_residual(vel, cfg.periodic_time)

    s_thr = _robust_threshold(sres, cfg.spatial_k)
    t_thr = _robust_threshold(tres, cfg.temporal_k)
    spatial_flag = sres > s_thr
    temporal_flag = tres > t_thr

    # Chronic = spatially wrong in too many frames.
    frames_off = spatial_flag.sum(axis=0)
    chronic = mask & (frames_off >= cfg.chronic_fraction * T)

    # Only exclude chronic voxels that stay connected to the exterior; excluding
    # an interior voxel would create an enclosed cavity (topology violation).
    # The rest are kept in the lumen and inpainted at *every* frame instead.
    exclude, reclaim = _topology_safe_exclusion(chronic, mask)
    updated_mask = mask & ~exclude

    # Drop edge/corner-attached speckle so the lumen is a single face-connected
    # component (done before cavity filling; removing surface speckle cannot
    # create an enclosed hole).
    if cfg.keep_largest_component:
        islands_removed = updated_mask & ~_largest_component(updated_mask)
        updated_mask = updated_mask & ~islands_removed
    else:
        islands_removed = np.zeros_like(updated_mask)

    # Optionally fill holes already present in the input segmentation so the
    # output mask is fully solver-ready.  Filled voxels get inpainted velocity.
    cavities_in = _enclosed_cavities(mask)
    if cfg.fill_preexisting_cavities:
        filled = _enclosed_cavities(updated_mask)
        updated_mask = updated_mask | filled
    else:
        filled = np.zeros_like(updated_mask)

    # (T, nz, ny, nx) lumen mask after exclusion, replicated over time.
    keep_mask = np.broadcast_to(updated_mask, vel.shape[:4]).copy()

    # Transient bad voxel-frames (spatial OR temporal flag), plus every frame of
    # the kept-but-chronically-bad ("reclaimed") interior voxels and any newly
    # filled cavity voxels.
    replace = keep_mask & (spatial_flag | temporal_flag)
    replace |= np.broadcast_to(reclaim | filled, vel.shape[:4])

    cleaned = inpaint(vel, keep_mask, replace, cfg)
    cleaned[~keep_mask] = 0.0  # excluded voxels and outside-lumen -> zero

    stats = {
        "n_lumen": int(mask.sum()),
        "n_chronic_detected": int(chronic.sum()),
        "n_excluded": int(exclude.sum()),
        "n_interior_reclaimed_inpainted": int(reclaim.sum()),
        "n_preexisting_cavity_voxels": int(cavities_in.sum()),
        "n_cavities_filled": int(filled.sum()),
        "n_speckle_voxels_removed": int(islands_removed.sum()),
        "n_voxelframes_replaced": int(replace.sum()),
        "spatial_threshold": s_thr,
        "temporal_threshold": t_thr,
        "excluded_on_boundary_pct": _boundary_pct(exclude, mask),
        "enclosed_cavities_after": int(_enclosed_cavities(updated_mask).sum()),
        "lumen_components_after": int(
            ndimage.label(updated_mask, structure=_FACE)[1],
        ),
    }
    return FilterResult(cleaned, updated_mask, exclude, replace, stats)


def _boundary_pct(sel, mask):
    if sel.sum() == 0:
        return float("nan")
    nmask = ndimage.convolve(mask.astype(np.float32), _FOOT, mode="constant")
    boundary = mask & (nmask < 26)
    return round(100.0 * (sel & boundary).sum() / sel.sum(), 1)


# FIXME: change hardcoded strings
def write_outputs(result: FilterResult, template: pv.ImageData, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = result.mask.shape
    for t in range(result.velocity.shape[0]):
        frame = template.copy()
        frame.cell_data.clear()
        frame.cell_data["velocity"] = result.velocity[t].reshape(-1, 3)
        frame.save(out_dir / f"UM8_filtered_{t:03d}.vti")
    mask_img = template.copy()
    mask_img.cell_data.clear()
    mask_img.cell_data["mask"] = result.mask.reshape(-1).astype(np.float64)
    mask_img.save(out_dir / "UM8_filtered_mask.vti")

    report = out_dir / "filter_report.txt"
    lines = ["4D-flow filter report", "=" * 40]
    lines += [f"{k}: {v}" for k, v in result.stats.items()]
    report.write_text("\n".join(lines) + "\n")


def main():

    DATA_DIR = Path(
        "/Users/bkhardy/Developer/aorta-model-utilities",
    )
    N_TIMEPOINTS = 21

    cfg = FilterConfig()
    print("Loading stack...")
    vel, mask, template = load_stack(DATA_DIR, N_TIMEPOINTS)
    print(f"  lumen voxels: {int(mask.sum())},  frames: {vel.shape[0]}")

    print("Filtering...")
    result = filter_4dflow(vel, mask, cfg)
    for k, v in result.stats.items():
        print(f"  {k}: {v}")

    out_dir = Path(__file__).parent / "output"
    print(f"Writing outputs to {out_dir} ...")
    write_outputs(result, template, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
