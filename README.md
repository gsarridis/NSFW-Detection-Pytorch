# NSFW-Detection-Pytorch

## Description
In this repository you can find an effective CNN model for detecting NSFW content in images. It achieves a 97.70% accuracy on the [pornography-2k](https://recod.ai/code-data/#porno) dataset. Apart from the provided code you need to reproduce the model, you can also use this model as python module (see below). 

## Usage
First install the python module:
```
pip install --upgrade nsfw_detector
```
Then use the model by:
```
#import the model
from nsfw_detector.model import Model

# initialize the model
net = Model()

# make a prediction
output = net.predict(<imagepath>)

# make multiple predictions
output = net.predict([<imagepath>, <imagepath>])

```
The model's output is as follows:
{'image_path': {'Label': SFW or NSFW , 'Score': 0 to 1 }}

## Reproducability
Using the provided code you can retrain the model. Regarding the training/testing data, the corresponding splits are provided in folder "splits". You can find and download the necessary data from [here](https://recod.ai/code-data/#porno) (this is a private dataset, due to its content and you should send a request to gain access - find more info in the link) and [here](https://archive.org/download/NudeNet_classifier_dataset_v1). After downloading the 2 datasets, you should run the
```
python script1_extract_image_frames.py
```
to extract the frames from the videos of the pornography-2k dataset and then you can run the 
```
bash train_exps.sh 
```
to start training the model.

## Results Visualisation
As you can see in the following attention maps, the model exhibits high accuracy in terms of the image regions it focuses. (Censored images) <br />
![](./attention_maps/1.png?raw=true) ![](./attention_maps/2.png?raw=true) ![](./attention_maps/3.png?raw=true) ![](./attention_maps/4.png?raw=true)

