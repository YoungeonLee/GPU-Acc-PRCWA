import numpy as np
import torch
from matplotlib import pyplot as plt
import scipy.io

import torcwa
# We need to use our own materials for this, not the ones they provide
#import Materials

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False # Set to True if using RTX 3090 or above
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cpu')


