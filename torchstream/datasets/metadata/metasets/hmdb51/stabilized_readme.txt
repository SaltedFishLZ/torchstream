
There are 3 variables in each mat file

mask_out -  a [frame height] x [frame width] x [ # frames] matrix, the mask image for every frame (only pixels with 1 are valid, the rest is background)

xtform_out  -  3 x 3 x [ # frames] matrix, the image transformation for each frame from frame t-1 to frame t. 
The first one is by default eye(3). Each following one contains the transformation from the previous frame to the current frame.

BB - the bounding box of the final clip on the image plane. 

