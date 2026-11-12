from flowprep.nrrd_to_vti import nrrd_to_vti


def main() -> None:
    # NOTE: These are the arguments you could write in the commmand line! You can also just import the function directly
    # (as I'm doing below) instead of dealing with the command line interface.

    fh_path = "/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_FH/505 MR DelRec - 4D PC_FH - 42 frames Volume Sequence by InstanceNumber {:d}.nrrd"
    ap_path = "/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_AP/504 MR DelRec - 4D PC_AP - 21 frames Volume Sequence by TriggerTime {:d}.nrrd"
    rl_path = "/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/PC_RL/503 MR DelRec - 4D PC_RL - 21 frames Volume Sequence by TriggerTime {:d}.nrrd"

    seg_path = "/Users/bkhardy/Library/CloudStorage/Dropbox-UniversityofMichigan/Brandon Hardy/fsi_ad/AD_model2/data/4d_flow_nrrd/Segmentation.nrrd"

    out_dir = "./UM8_output"  # NOTE: make sure this exists first!!

    nrrd_to_vti(
        image_patterns=[
            rl_path,
            ap_path,
            fh_path,
        ],
        seg_path=seg_path,
        frames=(0, 20),
        signs=(+1, +1, +1),
        to_ras=False,
        out_prefix="UM8_velocity",
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
