#!/bin/bash
echo "Uploading checkpoints"

rsync -avz --exclude "*.bash" ./ zhen@blaze:video-acc/pretrained_models
