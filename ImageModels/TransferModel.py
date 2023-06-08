import torch
from torch import nn

class Model(nn.Module):
    def __init__(self,model_arch,num_classes=10,weights="IMAGENET1K_V1",device="cuda"):
        super().__init__()
        self.model_arch=model_arch
        self.device=device
        self.num_classes=num_classes
        self.model=torch.hub.load("pytorch/vision", self.model_arch, weights=weights).to(self.device)
        
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in self.model.parameters():
            param.requires_grad = False
            
        layers=[n for n, _ in self.model.named_children()]
        last_layer= layers[-1]
        #print (last_layer)
        
        if last_layer=="fc": #Resnet and shufflenet
            self.in_features=(self.model).fc.in_features
            self.model.fc= torch.nn.Sequential( 
                torch.nn.Linear(in_features=self.in_features,out_features=128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=128,
                                out_features=self.num_classes, # same number of output units as our number of classes
                                bias=True)).to(self.device)
            
        elif last_layer=="classifier": #mobileNet and efficientnet and convnext
            self.in_features=(self.model).classifier[-1].in_features
            self.model.classifier[-1]= torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=self.in_features,out_features=128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=128,
                                out_features=self.num_classes, # same number of output units as our number of classes
                                bias=True)).to(self.device)     
            
        elif last_layer=="heads": #VIT_16
            #self.in_features=(self.model).heads.in_features
            self.model.heads= torch.nn.Sequential(
                torch.nn.Linear(in_features=768,out_features=128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=128,
                                out_features=self.num_classes, # same number of output units as our number of classes
                                bias=True)).to(self.device)        
        
    def forward(self,x):
        y=self.model(x)
        return (y)
