import sys
sys.path.append('train')
sys.path.append('.')

import utils
from utils import train_model, get_model_name

from mnist_models import TwoLayerNN, MLP, CNN, MLPBN, ConvNet, LeNet, LeNet5

import torch

for m in {TwoLayerNN(), MLP(), CNN(), MLPBN(), ConvNet(), LeNet(), LeNet5()}:
    mname = get_model_name(m)
    batchSize = 100
    loss_fn = torch.nn.CrossEntropyLoss()
    learningRate = 0.001

    optimizer = torch.optim.Adam(m.parameters(), lr = learningRate)

    num_epochs = 10

    train_model(m, loss_fn, batchSize, utils.trainset, utils.valset, optimizer, num_epochs)

    torch.save(m, 'checkpoints/' + mname)