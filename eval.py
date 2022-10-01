import sys
sys.path.append('.')
import utils
from utils import attack_model, get_model_name

import torch
import os

checkpoints = os.listdir('checkpoints')
for checkpoint in checkpoints:
    m = torch.load('checkpoints/' + checkpoint)
    mname = get_model_name(m)
    batchSize = 100
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 40

    print(mname)

    attack_model(m, loss_fn, batchSize, utils.valset, num_epochs)