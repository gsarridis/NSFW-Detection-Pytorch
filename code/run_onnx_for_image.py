import argparse

import numpy as np
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--image_path', type=str, default='../data/pornography-2k/images/porn/<image_id>.jpg', help="Directory of the test image.")
    parser.add_argument('--model_path', type=str, default='../results/models/2022_06_15_14_03_22.onnx', help="Model's directory.")
    # Parse the argument
    args = parser.parse_args()
    return args

# sigmoid function
def sig(x):
 return 1/(1 + np.exp(-x))

# loads an image and performs the transformations 
def load_image(path):
    # img = io.imread(path)
    img = Image.open(path)
    size = 240
    transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    img = transform(img)
    img = img[None,:,:,:]
    return img

# converts a torch tensor to numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# predicts if an image is disturbing or not
def predict(image_path, model):
    '''
    Args:
        image_path (str): Input image's directory
    Returns:
        list: first element: anotation (nsfw, sfw) and second element: (0,1) value, 0 means sfw (i.e., 0%) and 1 means nsfw (i.e., 100%)
    '''
    img = load_image(image_path)
    ort_session = onnxruntime.InferenceSession(model)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs) 
    l = []
    for im in ort_outs[0]:
        out_prob =  sig(im)[0]
        if out_prob>0.5:
            out_annotation = "nsfw"
        else:
            out_annotation = "sfw"
        l.append([out_annotation,out_prob])
    return l




def main():
    args = parse_args()
    # print the prediction for a sample image
    print (predict(args.image_path, args.model_path))

if __name__ == '__main__':
    main()
    