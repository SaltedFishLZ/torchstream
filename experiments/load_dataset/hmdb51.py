import os
import sys
sys.path.insert(1, '/rscratch/bernie/video-acc/torchstream-datasets')
import torch
import torchvision
import torchstream 
from torchstream import datasets
from torchvision import transforms


DATA_DIR = os.path.expanduser("/dnn/data/HMDB51/HMDB51-avi")

hmdb51_data = datasets.HMDB51(DATA_DIR, True)
print(hmdb51_data[2000])

dataloader = torch.utils.data.DataLoader(hmdb51_data,
                                          batch_size=1,
                                          shuffle=False)

# for i_batch, sample_batched in enumerate(dataloader):	
    

#     if i_batch == 20:
#     	print(sample_batched[0])
#     	print(sample_batched[1])
    
#     	break

# print(hmdb51_data.datapoints[0].ext)



