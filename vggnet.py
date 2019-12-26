import torch
import torch.nn as nn
import torch.nn.functional as F

VGG_TYPE = ["vgg11", "vgg13", "vgg16", "vgg19"]
VGG_CONFIG = {
    "vgg11" : [64,"m",128,"m",256,256,"m",512,512,"m",512,512,"m"],
    "vgg13" : [64,64,"m",128,128,"m",256,256,"m",512,512,"m",512,512,"m"],
    "vgg16" : [64,64,"m",128,128,"m",256,256,256,"m",512,512,512,"m",512,512,512,"m"],
    "vgg19" : [64,64,"m",128,128,"m",256,256,256,256,"m",512,512,512,512,"m",512,512,512,512,"m"],
}

def create_model(input):
    if input not in VGG_TYPE:
        print("Input not valid. Using vgg16 by default")
        input = "vgg16"
    
    layers = []
    config = VGG_CONFIG[input]
    input_dim = 3
    for v in config:
        if v == "m":
            layers.append(nn.MaxPool2d(2,2))
        else:
            layers.append(nn.Conv2d(input_dim, v, 3, padding=1))
            layers.append(nn.ReLU())
            input_dim = v

    

    return nn.Sequential(*layers)

class VGGNet(nn.Module):
    def __init__(self, model_type="vgg16"):
        super(VGGNet, self).__init__()

        # self.net = nn.Sequential(
        #     # channel-64
        #     nn.Conv2d(3, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
        #     # channel-128
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
        #     # channel-256
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
        #     # channel-512
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
        # )

        self.net = create_model(model_type)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 100),
        )

        self.setup_weights()
        

    def setup_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform(layer.weight)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
