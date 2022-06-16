import os

import torch
import torchvision
from torch import nn


def save_to_checkpoint(net, optimizer, epoch, path, loss, exp_name, step):
    """Saves a model

    Args:
        net (_type_): the model
        optimizer (_type_): the optimizer
        epoch (_type_): the current epoch
        path (_type_): the output directory
        loss (_type_): the current
        exp_name (_type_): the experiment id    
        step (_type_): the current step
    """ 
    # Additional information
    if optimizer == None:
        optim_state = None
    else:
        optim_state = optimizer.state_dict()
    torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim_state,
                'loss': loss,
                'exp_id': exp_name,
                }, os.path.join(path,exp_name+'.pt'))


def load_from_checkpoint(net,optimizer, path, ):
    """Loads a checkpoint model

    Args:
        net (_type_): the model
        optimizer (_type_): the optimizer (can be None)
        path (str): the path of the saved model

    Returns:
        net: the model
        optimizer: the optimizer
        start_info: info about the checkpoint (epoch, step)
        loss: the loss
        exp_name: the id of the checkpoint
    """
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if not optimizer == None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_info = { 'epoch': checkpoint['epoch'],
                    'step': checkpoint['step'] }
    loss = checkpoint['loss']
    exp_name = checkpoint['exp_id']
    print(f'Best Epoch: {start_info["epoch"]}')
    return net, optimizer, start_info, loss, exp_name


def ensure_reproducability():
    """Use this function to avoid randomness during training

    Returns:
        _type_: the generator
    """
    # disable convolution benchmarking to avoid randomness
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #seed torch
    torch.manual_seed(0)
    # generator for dataloader
    g = torch.Generator()
    g.manual_seed(0)
    return g


def load_model(name, classes=2, isbinary=True, pretrained=True):  
    """Initializes a CNN model

    Args:
        name (str): the name of the model to load
        classes (int, optional): the number of output classes. Defaults to 2.
        isbinary (bool, optional): If you want a binary output set it as True. Defaults to True.
        pretrained (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: the model
    """
    if isbinary:
        classes = 1
    #load the original model
    if name == 'efficientnet-b0':
        model = torchvision.models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,classes)
    elif name == 'efficientnet-b1':
        model = torchvision.models.efficientnet_b1(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,classes)
    elif name == 'efficientnet-b2':
        model = torchvision.models.efficientnet_b2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,classes)
    elif name == 'efficientnet-b3':
        model = torchvision.models.efficientnet_b3(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,classes)
    elif name == 'efficientnet-b4':
        model = torchvision.models.efficientnet_b4(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,classes)
    return model
