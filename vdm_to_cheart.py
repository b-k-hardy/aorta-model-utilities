import meshio as io
import numpy as np
import pyvista as pv


def write_dfile(filename: str, array: np.ndarray) -> None:
    shape = array.shape
    s1 = shape[0]
    s2 = shape[1]
    np.savetxt(filename, array, fmt="%30.15f", header=f"{s1}\t{s2}", comments="")


def write_xfile(filename: str, points: np.ndarray) -> None:
    np.savetxt(
        filename,
        points,
        fmt="%30.15f",
        delimiter="\t",
        header=f"{points.shape[0]}\t{points.shape[1]}",
        comments="",
    )


def write_tfile(filename, elems, points) -> None:
    np.savetxt(
        filename,
        elems + 1,
        fmt="%i",
        delimiter="\t",
        header=f"{elems.shape[0]}\t{points.shape[0]}",
        comments="",
    )


def write_mesh(filename, points, elems) -> None:
    write_xfile(f"{filename}_FE.X", points)
    write_tfile(f"{filename}_FE.T", elems, points)


def read_mesh(path) -> tuple[np.ndarray, np.ndarray, str]:
    # Load mesh
    xyz = np.loadtxt(path + "_FE.X", skiprows=1)
    ien = np.loadtxt(path + "_FE.T", skiprows=1, dtype=int) - 1

    return xyz, ien, "triangle"


def mesh_to_vtu(mesh_path, out_name, displacement, elem=None) -> None:
    xfile, tfile, elem = read_mesh(mesh_path)

    io.write_points_cells(out_name, xfile, {elem: tfile}, point_data={"Displacement": displacement})


def vdm_to_cheart(vdm_file: str, out_name: str, *, check: bool = False) -> None:
    """Convert a VDM mesh to CHeart format.

    Args:
        vdm_file (str): path to the VDM file.
        out_name (str): base name for the output files.
        check (bool, optional): whether to perform a round-trip check. Defaults to False.

    """
    vdm = pv.read(vdm_file)
    points = vdm.points
    faces = vdm.irregular_faces
    n_triangles = len(faces)
    elems = np.zeros((n_triangles, 3), dtype=int)
    for i in range(n_triangles):
        elems[i, :] = faces[i]

    write_mesh(out_name, points, elems)
    write_dfile(f"{out_name}_disp.D", vdm.point_data["Displacement"])

    if check:
        mesh_to_vtu(out_name, f"{out_name}_round_trip.vtu", vdm.point_data["Displacement"], elem="triangle")


def main() -> None:
    """Run VDM to CHeart converter.

    Alternatively, you can import vdm_to_cheart() and just use it in your own code.
    """
    # Replace with your actual VDM file path
    vdm_file = "/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/VDM/UM8_VDM-D.vtp"

    # Set output name and determine if you want a check
    vdm_to_cheart(vdm_file, "UM8_VDM", check=True)


if __name__ == "__main__":
    main()
