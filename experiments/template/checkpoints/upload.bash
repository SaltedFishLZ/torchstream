#!/bin/bash
echo "Uploading checkpoints"

rsync -avz --exclude "*.bash" \
    ./ \
    zhen@a18.millennium.berkeley.edu:video-acc/pretrained_models
