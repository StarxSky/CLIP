import os 
import torch
import warnings

from .clip import *
from torch.backends import mps
from pkg_resources import packaging



bar = "="
version = torch.__version__


if mps.is_available() :
    device = torch.device('mps')
elif torch.cuda.is_available() :
    device = torch.device('cuda')
else :
    device = torch.device('cpu')




print(f'{bar*10}Device INFO{bar*10}')
if packaging.version.parse(version) == packaging.version.parse("1.12.1") or packaging.version.parse("1.12.0") :
    warnings.warn("This PyTorch version 1.12.1 or higher is Support Metal GPU Boost!")

print(f'PyTorch Version :{version}')
print(f'Device :{device}')
print(bar*31)
