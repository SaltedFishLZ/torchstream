# Video Classification Datasets API for PyTorch

This part is meant to create an unified PyTorch dataset API for different datasets as well as different video data representations. You can use it as a Python package.

We will try our best to make it support more possiblities like different input data format (both video or sliced image sequences), different data representation (i.e., you can store the entire video in memory or in image sequences on disks)

## Dependency

* Python 3 (>=3.5)

* Python Packages

    All python packages for this project will be included in \<ROOT\>/requirement.txt
    * PyTorch 1.0
    * OpenCV for Python (opencv-python)


## Structure

* dataset.py

    This is the main PyTorch API for users. It will act as normal PyTorch Dataset class.
    
    It depends on the following modules:
    * metadata.py

        This part handles the meat-data part (e.g., file path, annotation) of different datasets.

    * video.py

        This part handles video processing (e.g., video decoding, slicing, image sequence read/write).

* transform.py

    Additional transformation functions for videos in PyTorch.


## For Developers

If you want to contribute to this package, you can read the package structure mentioned above and review the codes. After that, let's begin coding!

