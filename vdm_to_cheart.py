import meshio as io
import numpy as np
import pyvista as pv


def write_dfile(fname, array):  # TODO check if name finish in D or not
    shape = array.shape
    s1 = shape[0]
    s2 = shape[1]
    np.savetxt(fname, array, fmt="%30.15f", header=f"{s1}\t{s2}", comments="")


def write_xfile(fname, pts):  # TODO check if the extension is correct, if not add it
    np.savetxt(
        fname,
        pts,
        fmt="%30.15f",
        delimiter="\t",
        header=f"{pts.shape[0]}\t{pts.shape[1]}",
        comments="",
    )


def write_tfile(fname, elems, pts):
    np.savetxt(
        fname,
        elems + 1,
        fmt="%i",
        delimiter="\t",
        header=f"{elems.shape[0]}\t{pts.shape[0]}",
        comments="",
    )


def write_mesh(fname, pts, elems):
    write_xfile(f"{fname}_FE.X", pts)
    write_tfile(f"{fname}_FE.T", elems, pts)


def read_mesh(path, element=None):
    # Load mesh
    xyz = np.loadtxt(path + "_FE.X", skiprows=1)
    ien = np.loadtxt(path + "_FE.T", skiprows=1, dtype=int) - 1
    try:
        bfile = np.loadtxt(path + "_FE.B", skiprows=1)
    except:
        bfile = np.array([])

    # ien, element = get_element_type(ien, element=element, bfile=bfile)

    return xyz, ien, "triangle"


def mesh_to_vtu(mesh_path, out_name, elem=None):
    X, T, elem = read_mesh(mesh_path, element=elem)

    io.write_points_cells(out_name, X, {elem: T})


def main():

    CHECK = False

    vdm_file = "/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/VDM/UM8_VDM-D.vtp"  # Replace with your actual VDM file path
    vdm = pv.read(vdm_file)
    points = vdm.points
    faces = vdm.irregular_faces
    n_triangles = len(faces)
    elems = np.zeros((n_triangles, 3), dtype=int)
    for i in range(n_triangles):
        elems[i, :] = faces[i]

    write_mesh("UM8_VDM", points, elems)
    write_dfile("UM8_disp.D", vdm.point_data["Displacement"])

    if CHECK:
        mesh_to_vtu("UM8_VDM", "UM8_VDM_round_trip.vtu", elem="triangle")


if __name__ == "__main__":
    main()
