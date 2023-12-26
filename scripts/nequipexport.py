import torch
import sys
from collections import OrderedDict
checkpoint=torch.load(sys.argv[1],map_location='cpu')
out=OrderedDict()

module_conut=next(iter(checkpoint["state_dict"])).count("module")
for k,v in  checkpoint['state_dict'].items():
    newk="func."+k[len("module.") * abs(module_conut)+len("model.") :]
    out[newk]=v

torch.save(out,sys.argv[2])