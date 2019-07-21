#!/bin/bash
echo "Downloading logs"

rsync -avz --exclude "*.bash" zhen@blaze:video-acc/training_logs/ ./
