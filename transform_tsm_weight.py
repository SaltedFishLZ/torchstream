import sys
import glob

import tqdm
import torch

import utils

def transform_tsm_weight(state_dict):

    model_state_dict = state_dict["state_dict"]

    old_keys = list(model_state_dict.keys())
    for key in old_keys:
        val = model_state_dict[key]
        new_key = key.replace("module.", "")
        if "conv1.net" in new_key:
            new_key = new_key.replace('conv1.net', 'conv1.conv')
        if "new_fc" in new_key:
            new_key = new_key.replace('new_fc', 'base_model.fc.fc')
        del model_state_dict[key]
        model_state_dict[new_key] = val

    del state_dict["state_dict"]
    state_dict["model_state_dict"] = model_state_dict

    if "optimizer" in state_dict:
        optimizer_state_dict = state_dict["optimizer"]
        del state_dict["optimizer"]
        state_dict["optimizer_state_dict"] = optimizer_state_dict

    return state_dict

if __name__ == "__main__":
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        checkpoints = glob.glob(dir_path+"/*/*/*.pth")
        
        for checkpoint in tqdm.tqdm(checkpoints):
            # print(40*"##")
            # print(checkpoint)
            state_dict = torch.load(checkpoint)
            # print(40*"##")
            # print(state_dict.keys())
            state_dict = transform_tsm_weight(state_dict)
            # print(state_dict.keys())
            # print(state_dict["model_state_dict"].keys())
            torch.save(state_dict, checkpoint)
    else:
        print("cmd <pth path>")
