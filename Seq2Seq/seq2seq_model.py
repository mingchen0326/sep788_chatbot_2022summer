# -*- coding: UTF-8 -*-
# Sequence to Sequence Model

# requires
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

