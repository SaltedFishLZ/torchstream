# Video Classification Datasets API for PyTorch

This part is meant to create an unified PyTorch dataset API for different datasets as well as different video data representations. You can use it as a Python package.

We will try our best to make it support more possiblities like different input data format (both video or sliced image sequences), different data representation (i.e., you can store the entire video in memory or in image sequences on disks)

**NOTE**:  
Currently, we only support images in `.jpg` format.

## Dependency

* Python 3 (>=3.5)

* Python Packages  
All python packages for this project will be included in `\<ROOT\>/requirement.txt`. Here, we will list packages required by this `vdataset` package.  
  * PyTorch 1.0
  * OpenCV for Python (opencv-python)

## Structure

### Overview

* dataset  
This is the main PyTorch API for users. It will act as normal PyTorch Dataset class.  
* metadata  
This part handles the meat-data part (e.g., file path, annotation) of different datasets.
* video  
This part handles video processing (e.g., video decoding, slicing, image sequence read/write).
* transform  
Additional transformation functions for videos in PyTorch.

The dependency graph is shown below:

```sequence
metadata.py -> dataset.py  
video.py -> dataset.py
```

### dataset

This module is the top module.

### metadata

### Video Array

### Image Sequence
    The folder shall looks like this:
    video path
    ├── frame 0
    ├── frame 1
    ├── ...
    └── frame N
    NOTE: Following the "do one thing at once" priciple, we only deal with 1 
    data type of 1 data modality in 1 collector object.

## For Developers

If you want to contribute to this package, you can read the package structure mentioned above and review the codes. After that, let's begin coding~