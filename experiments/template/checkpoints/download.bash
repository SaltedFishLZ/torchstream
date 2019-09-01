#!/bin/bash
echo "Downloading checkpoints"

# if the source server delete something, local side will delete too
rsync -avz --exclude "*.bash" zhen@a18.millennium.berkeley.edu:video-acc/pretrained_models/ ./ --delete
