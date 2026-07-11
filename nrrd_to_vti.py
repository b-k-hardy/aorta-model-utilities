import nrrd
import numpy as np
import pyvista as pv


def assemble_velocity_vti(paths, segmentation, timestep, signs=(+1, +1, +1), to_ras=False, out_prefix="vel"):
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


def main() -> None:

    # TODO (for Brandon): incorporate Slicer scripting (manual NRRD export for now)

    # path to segmentation that has already been exported into 4d flow grid
    segmentation_path = (
        "/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/"
        "data/4d_flow_nrrd/Segmentation.v2.nrrd"
    )
    seg_data, _ = nrrd.read(segmentation_path)

    # Number of timesteps in 4d flow DICOM series.
    # NOTE: If you don't know this you can find it in 3D slicer in the top left where it says "Sequence Browser"
    N_TIMESTEPS = 21

    for t in range(N_TIMESTEPS):
        print(f"Processing time step {t}...")
        fh_path = f"/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_FH/505 MR DelRec - 4D PC_FH - 42 frames Volume Sequence by InstanceNumber {21 + t}.nrrd"
        ap_path = f"/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_AP/504 MR DelRec - 4D PC_AP - 21 frames Volume Sequence by TriggerTime {t}.nrrd"
        rl_path = f"/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_RL/503 MR DelRec - 4D PC_RL - 21 frames Volume Sequence by TriggerTime {t}.nrrd"

        image_paths = {"fh": fh_path, "ap": ap_path, "rl": rl_path}

        assemble_velocity_vti(image_paths, seg_data, t, signs=(+1, +1, +1), to_ras=False, out_prefix="UM8_velocity")


if __name__ == "__main__":
    main()
