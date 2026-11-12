import argparse
import sys
from pathlib import Path

import nrrd
import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from tqdm import tqdm


def assemble_velocity_vti_step(
    paths: dict[str, str],
    segmentation: NDArray,
    timestep: int,
    signs: tuple[int, int, int] = (+1, +1, +1),
    *,
    to_ras: bool = False,
    out_prefix: str = "vel",
    out_dir: str | None = None,
):
    """Convert nrrd files exported from 3D Slicer to vti.

    paths: dict {'rl': ..., 'ap': ..., 'fh': ...}  -> NRRD per component
    signs: (s_rl, s_ap, s_fh) in {+1,-1}; verify empirically (see below)
    to_ras: if True, convert geometry AND velocity LPS->RAS (flip x,y)
    Each NRRD is one velocity component, 3D (nx,ny,nz)
    """
    # read headers and data
    # mask velocity and convert to m/s
    data, hdrs = {}, {}
    for k, p in paths.items():
        data[k], hdrs[k] = nrrd.read(p)
        data[k] *= segmentation / 100  # converting to m/s

    # --- geometry from ONE header; assert the others match ---
    h = hdrs["rl"]
    assert h.get("space", "").replace("-", " ").lower().startswith("left posterior"), (
        f"expected LPS, got {h.get('space')}"
    )
    M = np.array(h["space directions"][:3], float).T  # columns = per-axis world vectors
    org = np.array(h["space origin"], float)
    spacing = np.linalg.norm(M, axis=0)
    direction = M / spacing  # unit columns (orthonormal for rigid oblique)

    for k in paths:
        assert np.allclose(np.array(hdrs[k]["space directions"][:3], float).T, M), f"{k}: geometry mismatch"
        assert np.allclose(np.array(hdrs[k]["space origin"], float), org), f"{k}: origin mismatch"
        assert data[k].shape == data["rl"].shape, f"{k}: shape mismatch"

    s_rl, s_ap, s_fh = signs

    # optional LPS -> RAS (flip L and P): applies identically to geometry and vectors
    F = np.diag([-1e-3, -1e-3, 1e-3]) if to_ras else 1e-3 * np.eye(3)
    org_w, dir_w = F @ org, F @ direction

    sl = (Ellipsis,)
    rl, ap, fh = data["rl"][sl], data["ap"][sl], data["fh"][sl]
    nx, ny, nz = rl.shape

    # world-frame components in LPS order — NOT rotated by M
    vx = (s_rl * rl).flatten(order="F")  # L
    vy = (s_ap * ap).flatten(order="F")  # P
    vz = (s_fh * fh).flatten(order="F")  # S
    vec = np.column_stack([vx, vy, vz])  # (N,3), x-fastest to match VTK ordering
    if to_ras:
        vec = vec @ F.T  # flip vx, vy

    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    grid.origin = tuple(org_w)
    grid.spacing = tuple(spacing)
    grid.direction_matrix = dir_w  # needs VTK 9+/recent ParaView
    grid.cell_data["velocity"] = vec
    grid.cell_data.active_vectors_name = "velocity"
    if out_dir:
        grid.save(f"{out_dir}/{out_prefix}_t{timestep:03d}.vti")
    else:
        grid.save(f"{out_prefix}_t{timestep:03d}.vti")


def nrrd_to_vti(
    image_patterns,
    seg_path,
    frames: tuple[int, int],
    signs=(+1, +1, +1),
    *,
    to_ras=False,
    out_prefix="vel",
    out_dir=None,
):
    rl_pat, ap_pat, fh_pat = image_patterns  # RL, AP, FH order, matching the CLI help
    seg_data, _ = nrrd.read(seg_path)
    for t in tqdm(range(frames[0], frames[1] + 1)):
        # print(f"Processing time step {t}...")
        paths_t = {
            "rl": Path(rl_pat.format(t)),
            "ap": Path(ap_pat.format(t)),
            "fh": Path(fh_pat.format(t)),
        }
        assemble_velocity_vti_step(
            paths_t,
            seg_data,
            t,
            signs=signs,
            to_ras=to_ras,
            out_prefix=out_prefix,
            out_dir=out_dir,
        )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Convert 4D Flow NRRD exported from 3D Slicer to VTI")
    p.add_argument(
        "--image_path_pattern",
        "-ipp",
        type=str,
        nargs=3,
        default=None,
        required=True,
        help="Paths to NRRD files for RL, AP, FH components (in that order). E.g., data/file{:02d}.nrrd",
    )
    p.add_argument(
        "--frames",
        "-f",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        required=True,
        help="First and last frame numbers (inclusive) to process",
    )
    p.add_argument("--seg_path", type=str, default=None, required=True, help="Path to segmentation NRRD file")

    p.add_argument(
        "--signs",
        "-s",
        type=int,
        nargs=3,
        default=(+1, +1, +1),
        help="Signs for RL, AP, FH components (in that order)",
    )
    p.add_argument(
        "--to_ras",
        "-r",
        action="store_true",
        help="Convert geometry and velocity from LPS to RAS",
    )
    p.add_argument("--out_prefix", "-o", type=str, default="vel", help="Prefix for output VTI files")

    p.add_argument(
        "--out_dir",
        "-od",
        type=str,
        default=None,
        help="Output directory for VTI files. DIRECTORY MUST EXIST. (default: current directory)",
    )

    args = p.parse_args(argv)

    # TODO potentially incorporate Slicer scripting (manual NRRD export for now)

    nrrd_to_vti(
        args.image_path_pattern,
        args.seg_path,
        tuple(args.frames),
        signs=args.signs,
        to_ras=args.to_ras,
        out_prefix=args.out_prefix,
        out_dir=args.out_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
