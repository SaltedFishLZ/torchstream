#!/bin/bash
echo "Uploading checkpoints"

rsync -avz --exclude "*.bash" ./ zhen@a18:video-acc/pretrained_models
