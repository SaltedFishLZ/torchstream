#!/bin/bash
echo "Updating logs"

rsync -avz --exclude "*.bash" ./ zhen@blaze:video-acc/training_logs
