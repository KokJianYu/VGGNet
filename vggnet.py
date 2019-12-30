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
            layers.append(nn.BatchNorm2d(v))
            input_dim = v

    

    return nn.Sequential(*layers)

class VGGNet(nn.Module):
    def __init__(self, model_type="vgg16", num_classes = 1000, is_training = True):
        super(VGGNet, self).__init__()

        self.net = create_model(model_type)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def setup_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        for layer in self.classifier_train:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
