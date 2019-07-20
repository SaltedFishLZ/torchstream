#!/bin/bash
echo "Syncing checkpoints"

rsync -avz ./ zhen@blaze:video-acc/pretrained_models
