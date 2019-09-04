source ../../venv_video/bin/activate

# data prallel
sudo mkdir -p logs/tsm/sthsthv1/resnet50/8x224x224/train/finetune/
python train.py \
    configs/tsm/sthsthv1/resnet50/8x224x224/train/finetune/kinetics0_lr5e-3_p9e-1_step20_b64_j64_e50.json \
    > >(sudo tee logs/tsm/sthsthv1/resnet50/8x224x224/train/finetune/kinetics0_lr5e-3_p9e-1_step20_b64_j64_e50.log)

