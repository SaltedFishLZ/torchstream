import sys
import glob

import tqdm
import torch

def remove_module_in_key(state_dict):

    model_state_dict = state_dict["model_state_dict"]

    old_keys = list(model_state_dict.keys())
    for key in old_keys:
        val = model_state_dict[key]
        new_key = key.replace("module.", "")
        del model_state_dict[key]
        model_state_dict[new_key] = val

    return state_dict

if __name__ == "__main__":
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        checkpoints = glob.glob(dir_path+"/*.pth")

        for checkpoint in tqdm.tqdm(checkpoints):
            # print(40*"##")
            # print(checkpoint)
            state_dict = torch.load(checkpoint)
            # print(40*"##")
            # print(state_dict.keys())
            state_dict = remove_module_in_key(state_dict)
            # print(state_dict.keys())
            # print(state_dict["model_state_dict"].keys())
            torch.save(state_dict, checkpoint)
    else:
        print("cmd <pth path>")
