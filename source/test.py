

import sys

import torch

from model import densenet
from model import adjustmentnet



model = densenet
model.load_state_dict(torch.load("model/chkpoint_003.pt"))
model.cuda()
model.eval()


adjustmnetnet.load_state_dict(torch.load("model/chkpoint_adj_000.pt"))
adjustmentnet.cuda()
adjustmentnet.eval()


imgin = sys.argv[1]


