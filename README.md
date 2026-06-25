# aorta-model-utilities

**NOTE:** This repository is *mirrored* to GitLab and the cardiac-biomechanics-lab group has access to this mirror. I primarily use GitHub for my own code development, so the mirroring is primarily a way to ensure that my code is accessible to anyone in the lab who needs these utilities in the future.

This repository contains useful utilities for setting up FSI TBAD models. The primary focus for now will be on managing 4D Flow data. This includes segmentation and filtering the raw data.

## Setup and `uv`

Two main things are required to run this pipeline: first, a segmentation of the MRI data that has been exported into the 4D flow image space. Second, `uv` must be installed, which you can find [at this link](https://docs.astral.sh/uv/). Basically, `uv` is a python package manager that automates a lot of the behind-the-scenes Python stuff to make sure that we're all running the exact same code and nothing breaks.

There are two ways I normally use `uv`. The first way is to just use it for everything; after cloning this repository, run `uv run <any_python_file>` in the terminal and then `uv` will automatically download any dependencies and generate a local "virtual" python environment (basically this just means that any required external packages are installed alongside this codebase). Then you continue using `uv run <script_name>` anytime you want to run code.

Alternatively, after cloning this repository, you could run `uv sync`, which will generate the virtual environment. You can then manually activate the virtual environment with `source .venv/bin/activate` and your terminal should change to reflect the activation. Next run `python3 <script_name>` whenever you want to run something. Practically there is very little difference between these methods, but some people (me) have trouble getting over the muscle memory of writing "python" in the terminal.

## Code Overview

In each Python file I've tried to make sure the code is well-commented and easy to follow, so I've just written out the high-level purpose of each script below.

### `nrrd_to_vti.py`

Before running this script, ensure that all the 4D Flow velocity fields are exported from 3D Slicer into NRRD format. Every velocity field has its own special problems so just let me know if you're starting a new export and I can help. Basically all we're doing here is converting NRRD format to VTI because Slicer doesn't natively export to modern VTI and Paraview can't read the NRRD format very well. Basically, we need a small communication layer to get the flow split data going.

### `4dflow_filter.py`

This should be run *after* you have the 4D Flow NRRD files exported to vti. It will load in all of the files and attempt to nicely clean up the velocity data so it's easier to view.

### `vdm_to_cheart.py`

Use this script to convert VDM files to .X, .T, and displacement.D files. This is important for the total FSI pipeline but not related to the 4D Flow data.
