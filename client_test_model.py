from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL 
from PIL import Image 
import os
import imageio.v2 as imageio

# NOTE: This is a hack to get around "User-agent" limitations when downloading MNIST datasets
#       see, https://github.com/pytorch/vision/issues/3497 for more information
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

pretrained_model = "/home/manita/Desktop/lenet_mnist_model.pth"
#use_cuda=True

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        print()
        return F.log_softmax(x, dim=1)

# Define what device we are using
#print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
print(model)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

#Testing function
def test(model, data):
    output = model(data)
    #final_pred = np.argmax(output)final_pred,
    return output

# Run test for each picture
n = len(os.listdir("/home/manita/Desktop/adv_picture"))
transform = transforms.Compose([transforms.ToTensor()])
"""test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),])),
        batch_size=1, shuffle=True)
"""
for i in range(n):
    image = "/home/manita/Desktop/adv_picture/" + "ad" + str(i+1) + ".jpg"
    image = Image.open(image).convert('L')
    image = transform(image).unsqueeze(1)
    #print(image.shape)
    #image = image.squeeze().detach().cpu()
    output  = test(model, image)
    #print(output)
    out = torch.nn.functional.softmax(output,dim=1)
    print(out)
    print(torch.argmax(out))
