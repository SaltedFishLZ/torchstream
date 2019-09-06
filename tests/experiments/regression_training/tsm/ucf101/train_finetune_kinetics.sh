source ../../venv_video/bin/activate

# data prallel
mkdir -p logs/tsm/ucf101/resnet50/8x224x224/train/finetune/
python -u train.py \
    configs/tsm/ucf101/resnet50/8x224x224/train/finetune/kinetics0_lr1e-3_p9e-1_step10_b64_j64_e100.json \
    > >(tee logs/tsm/ucf101/resnet50/8x224x224/train/finetune/kinetics0_lr1e-3_p9e-1_step10_b64_j64_e100.log)

# single node distributed data parallel
# NOTE:
# cannot use tee for multi-processing
sudo mkdir -p logs/tsm/ucf101/resnet50/8x224x224/train/finetune/
python -u train.py --distributed --nodes 1 \
    configs/tsm/ucf101/resnet50/8x224x224/train/finetune/kinetics0_lr1e-3_p9e-1_step10_b64_j64_e100.json