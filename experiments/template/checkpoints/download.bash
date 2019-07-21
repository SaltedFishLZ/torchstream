#!/bin/bash
echo "Downloading checkpoints"

rsync -avz --exclude "*.bash" zhen@blaze:video-acc/pretrained_models/ ./
