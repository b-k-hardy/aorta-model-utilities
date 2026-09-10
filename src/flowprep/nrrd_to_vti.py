import argparse
import sys

import nrrd
import numpy as np
import pyvista as pv
from numpy.typing import NDArray


def assemble_velocity_vti_step(
    paths: dict[str, str],
    segmentation: NDArray,
    timestep: int,
    signs: tuple[int, int, int] = (+1, +1, +1),
    *,
    to_ras: bool = False,
    out_prefix: str = "vel",
):
    """Convert nrrd files to vti.

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
    grid.save(f"{out_prefix}_t{timestep:03d}.vti")


def nrrd_to_vti(image_paths, seg_data, n_timesteps: int, signs=(+1, +1, +1), *, to_ras=False, out_prefix="vel"):

    for t in range(n_timesteps):
        print(f"Processing time step {t}...")

        image_paths = {"fh": image_paths[0], "ap": image_paths[1], "rl": image_paths[2]}

        assemble_velocity_vti_step(
            image_paths,
            seg_data,
            t,
            signs=(+1, +1, +1),
            to_ras=False,
            out_prefix="UM8_velocity",
        )


def main(argv=None) -> None:

    p = argparse.ArgumentParser(description="Convert 4D Flow NRRD exported from 3D Slicer to VTI")
    p.add_argument("--seg_path", type=str, default=None, help="Path to segmentation NRRD file")
    p.add_argument("--n-timesteps", "-nt", type=int, default=1, help="Number of timesteps in 4D Flow DICOM series")
    p.add_argument(
        "--image_paths",
        "-ip",
        type=str,
        nargs=3,
        default=None,
        help="Paths to NRRD files for RL, AP, FH components (in that order)",
    )

    args = p.parse_args(argv)

    # TODO potentially incorporate Slicer scripting (manual NRRD export for now)

    # path to segmentation that has already been exported into 4d flow grid
    seg_data, _ = nrrd.read(args.seg_path)

    nrrd_to_vti(
        args.image_paths,
        seg_data,
        args.n_timesteps,
        signs=(+1, +1, +1),
        to_ras=False,
        out_prefix="UM8_velocity",
    )

    # NOTE: these are sample paths that I'm hanging onto for reference (for now)
    # fh_path = f"/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_FH/505 MR DelRec - 4D PC_FH - 42 frames Volume Sequence by InstanceNumber {21 + t}.nrrd"
    # ap_path = f"/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_AP/504 MR DelRec - 4D PC_AP - 21 frames Volume Sequence by TriggerTime {t}.nrrd"
    # rl_path = f"/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_RL/503 MR DelRec - 4D PC_RL - 21 frames Volume Sequence by TriggerTime {t}.nrrd"

    return 0


if __name__ == "__main__":
    sys.exit(main())
