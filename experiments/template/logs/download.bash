#!/bin/bash
echo "Downloading logs"

# if the source server delete something, local side will delete too
rsync -avz --exclude "*.bash" zhen@blaze:video-acc/training_logs/ ./ --delete
