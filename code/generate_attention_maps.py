import os
import sys

sys.path.append("/home/gsarridis/Desktop/projects/NSFW-detection/")
os.chdir('/home/gsarridis/Desktop/projects/NSFW-detection/')

import argparse

import torch
from medcam import medcam
from torchvision import transforms
from tqdm import tqdm

import utils.loaders
from utils.nn_utils import *
from utils.nn_utils import load_model as lm


def arg_parser():
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--modelname', type=str, default='efficientnet-b1', help='Choose a model', choices=['efficientnet-b0','efficientnet-b1','efficientnet-b1-pornography-2k-pretrained','efficientnet-b2','efficientnet-b3','efficientnet-b4'])
    parser.add_argument('--modelpath', type=str, default='./results/models/2022_06_15_14_03_22.pt')
    parser.add_argument('--out_dir',type=str, default='./results/attention_maps', help='The directory to save the attention maps')
    parser.add_argument('--device', type=str, default='cuda:0')
    # Parse the argument
    args = parser.parse_args()
    return args



def test(net, loader, device):
    """passes forward all the test images in order to enable medcam to generate the attention maps

    Args:
        net (_type_): the model
        loader (_type_): dataloaders
        device (_type_): the device
    """
    net.eval()
    for (inputs, _, _) in tqdm(loader):
        inputs= inputs.to(device)
        # apply inverse transforms to get the original images.
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

        inv_tensor = invTrans(inputs)
        _ = net(inputs, raw_input=inv_tensor)
    return 

def main():
    # Create the parser
    args = arg_parser()
    print(args)

    # Set device
    device = torch.device(args.device)

    # initialize model 
    model = lm(args.modelname, classes=2, isbinary=True, pretrained=True)
    model.to(device)
    
    # load pretrained model
    model, _, _, _, _ = load_from_checkpoint(model,None, args.modelpath)
    model.to(device)

    # load data
    _, val_loader = utils.loaders.custom_nsfw_data_loader(batch=1)

    # initialize the attention map visualizer
    model = medcam.inject(model, output_dir=args.out_dir ,backend='gcam',  save_maps=True, label='best')
    
    # generate the attention maps
    test(net=model, loader=val_loader,device=device)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    main()
    
