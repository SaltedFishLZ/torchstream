source ../../venv_video/bin/activate

# data prallel
mkdir -p logs/tsm/hmdb51/resnet50/8x224x224/train/finetune/
python -u train.py \
    configs/tsm/hmdb51/resnet50/8x224x224/train/finetune/kinetics0_lr1e-3_p9e-1_step10_b64_j64_e100.json \
    > >(tee logs/tsm/hmdb51/resnet50/8x224x224/train/finetune/kinetics0_lr1e-3_p9e-1_step10_b64_j64_e100.log)
