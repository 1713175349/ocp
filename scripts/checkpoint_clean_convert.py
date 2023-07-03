import torch
import sys
from collections import OrderedDict
checkpoint=torch.load(sys.argv[1])
out={}


# for k,v in  checkpoint.items():
#     if k in ["state_dict","normalizers",'amp',"optimizer"]:
#         out[k]=v
        
for k,v in  checkpoint.items():
    if k in ["state_dict",'amp']:
        out[k]=v

torch.save(out,sys.argv[2])