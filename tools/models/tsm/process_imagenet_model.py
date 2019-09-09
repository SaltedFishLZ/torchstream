import sys
import torch

def process_resnets(model_state_dict):
    """
    Args:
        model_state_dict: source imagenet model state dict
    Return:
        wrapped state dict
    """
    dst_sd = {}

    old_keys = list(model_state_dict.keys())
    for key in old_keys:
        val = model_state_dict[key]
        
        new_key = key
        if "layer" in key:
            if ".conv1" in key:
                new_key = new_key.replace('.conv1', '.conv1.conv')
        elif "fc" in key:
            new_key = new_key.replace('fc', 'fc.fc')
        
        del model_state_dict[key]

        new_key = "base_model." + new_key
        model_state_dict[new_key] = val
    
    dst_sd["model_state_dict"] = model_state_dict
    return dst_sd


if __name__ == "__main__":
    if len(sys.argv) == 3:
        src = sys.argv[1]
        dst = sys.argv[2]
        model_state_dict = torch.load(src)

        checkpoint = process_resnets(model_state_dict)

        torch.save(checkpoint, dst)
    else:
        print("cmd src dst")