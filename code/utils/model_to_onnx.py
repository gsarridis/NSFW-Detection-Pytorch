import os
import sys

sys.path.append("/home/gsarridis/Desktop/projects/NSFW-detection/")
os.chdir('/home/gsarridis/Desktop/projects/NSFW-detection/')

import argparse
import torch
import torch.onnx
from nn_utils import load_from_checkpoint, load_model


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--modelpath', type=str, default='./results/models/2022_06_15_14_03_22.pt', help="Directory that the model you want to convert is placed.")
    parser.add_argument('--export_path', type=str, default='./results/models/porn2k_nudenet_effb1_v2.onnx', help="Output directory and filename.")

    # Parse the argument
    args = parser.parse_args()
    return args
 
#Function to Convert to ONNX 
def Convert_ONNX(model, export_path): 
    """Converts a pytorch model to onnx format

    Args:
        model (_type_): the pytorch model
        export_path (str): the output path
    """
    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, 240, 240, device="cuda")  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         export_path,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                        'modelOutput' : {0 : 'batch_size'}}
         ) 
    print(" ") 
    print('Model has been converted to ONNX') 

def main():
    args = parse_args()
    # initialize the model
    model = load_model('efficientnet-b1', classes=2, isbinary=True, pretrained=False)
    model.to('cuda:0')
    # load from checkpoint
    model, _, _, _, _ = load_from_checkpoint(model,None, args.modelpath)
    model.to('cuda:0')
    Convert_ONNX(model, args.export_path)

if __name__ == '__main__':
    main()
    







