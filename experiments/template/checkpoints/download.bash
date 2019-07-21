#!/bin/bash
echo "Syncing checkpoints"

rsync -avz --exclude "*.bash" zhen@blaze:video-acc/pretrained_models/ ./
