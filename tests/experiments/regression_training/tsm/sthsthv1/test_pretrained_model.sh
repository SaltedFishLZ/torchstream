cd checkpoints
bash download.bash
cd ..

source ../../venv_video/bin/activate
python test.py \
    configs/tsm/sthsthv1/resnet50/8x224x224/test/b64_j64.json \
    --weights checkpoints/tsm/sthsthv1/resnet50/8x224x224/kinetics0_lr1e-3_p9e-1_step10_b64_j64_e100.best.pth
