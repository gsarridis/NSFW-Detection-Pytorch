from nsfw_detector.model import Model

# initialize the model
net = Model()

# make a prediction
output = net.predict(<imagepath>)

# make multiple predictions
output = net.predict([<imagepath>, <imagepath>])
