import os
import pydload
import numpy as np
import onnxruntime
from .image_utils import load_images

# sigmoid function
def sig(x):
    return 1/(1 + np.exp(-x))

class Model:
    """
    Class for loading model and running predictions.
    For example on how to use take a look the if __name__ == '__main__' part.
    """

    nsfw_model = None

    def __init__(self):
        """
        model = Classifier()
        """
        url = "https://github.com/gsarridis/NSFW-Detection-Pytorch/releases/download/pretrained_models/2022_06_15_14_03_22.onnx"
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, ".NSFWModel/")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        model_path = os.path.join(model_folder, os.path.basename(url))

        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        self.nsfw_model = onnxruntime.InferenceSession(model_path)




    def predict(
        self,
        image_paths=[],
        batch_size=4,
        image_size=(240, 240)
    ):
        """
        inputs:
            image_paths: list of image paths or can be a string too (for single image)
            batch_size: batch_size for running predictions
            image_size: size to which the image needs to be resized
            categories: since the model predicts numbers, categories is the list of actual names of categories
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        loaded_images, loaded_image_paths = load_images(
            image_paths, image_size, image_names=image_paths
        )

        if not loaded_image_paths:
            return {}

        preds = []
        model_preds = []
        sigmoid_v = np.vectorize(sig)
        while len(loaded_images):
            _model_preds = self.nsfw_model.run(
                [self.nsfw_model.get_outputs()[0].name],
                {self.nsfw_model.get_inputs()[0].name: loaded_images[:batch_size]},
            )[0]
            _model_preds = sigmoid_v(_model_preds)

            model_preds = [*model_preds, *(np.transpose(_model_preds).tolist()[0])]
            t_preds = np.rint(_model_preds)
            t_preds = np.transpose(t_preds).astype(int).tolist()[0]
            preds = [*preds, *t_preds]

            loaded_images = loaded_images[batch_size:]


        images_preds = {}

        for i, loaded_image_path in enumerate(loaded_image_paths):
            if not isinstance(loaded_image_path, str):
                loaded_image_path = i

            images_preds[loaded_image_path] = {}
            if preds[i]> 0.5:
                images_preds[loaded_image_path] = { 'Label': 'NSFW', 'Score': model_preds[i]}
            else:
                images_preds[loaded_image_path] = { 'Label': 'SFW', 'Score': model_preds[i]}
        return images_preds


if __name__ == "__main__":
    m = Model()

    while 1:
        print(
            "\n Enter single image path or multiple images seperated by ; \n"
        )
        images = input().split(";")
        images = [image.strip() for image in images]
        print(m.predict(images), "\n")
