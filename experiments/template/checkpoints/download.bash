#!/bin/bash
echo "Downloading checkpoints"


SERVER_PREFIX="zhen@a18.millennium.berkeley.edu:video-acc"


    # if the source server delete something, local side will delete too
    rsync -avz --exclude "*.bash" \
        "${SERVER_PREFIX}/pretrained_models/"" \
        ./ \
        --delete
