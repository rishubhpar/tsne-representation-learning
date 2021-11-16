import torch
import torch.nn as nn


class feature_extractor(nn.Module):
    def __init__(self, ndi, ndo):
        super(feature_extractor, self).__init__()
        # Creates a feature extractor based on the number of dimensions for output
        self.layer1 = nn.Linear(ndi, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, ndo)

        # Create init function 
        self.init_layers() 

    def forward(self, x):
        y = self.layer1(x)
        y = nn.ReLU()(y)
        y = self.layer2(y)
        y = nn.ReLU()(y)
        # We don't need any activation function as this has to match the embeddings predicted by tsne, which are unconstrained
        output = self.layer3(y)

        return output

    def init_layers(self):
        # Weight values add correctly
        x = 23
    

class classifier_model(nn.Module):
    def __init__(self, ndi):
        super(classifier_model, self).__init__()
        self.layer1 = nn.Linear(ndi, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 10)  

        # Create init layers
        self.init_layers()

    def forward(self, x):
        y = self.layer1(x)
        y = nn.ReLU()(y)
        y = self.layer2(y)
        y = nn.ReLU()(y)
        y = self.layer3(y)
        y = nn.ReLU()(y)
        y = self.layer4(y)
        y = nn.ReLU()(y) 
        output = self.layer5(y) # No need to add softmax as it will already being added in the loss function

        return output

    def init_layers(self):
        # Initialize the layers weights appropriately
        x = 23


class conv_feature_extractor(nn.Module):
    def __init__(self, ndi, ndo):
        print("Training conv feature extractor  ..... ")
        super(conv_feature_extractor, self).__init__()
        
        self.layer1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Conv2d(6, 16, 5)
        self.layer3 = nn.Linear(16 * 5 * 5, ndo) 

        # Create init layers
        self.init_layers()

    def forward(self, x):
        y = self.layer1(x)
        y = self.pool(nn.ReLU()(y))
        y = self.layer2(y)
        y = self.pool(nn.ReLU()(y))
        y = torch.flatten(y, 1)
        output = self.layer3(y) # We don't need any activation function as this has to match the embeddings predicted by tsne, which are unconstrained
 
        return output

    def init_layers(self):
        # Initialize the layers weights appropriately
        x = 23

class conv_classifier_model(nn.Module):
    def __init__(self, ndi):
        print("Training conv classifier  ..... ")
        super(conv_classifier_model, self).__init__()
        
        self.layer1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Conv2d(6, 16, 5)
        self.layer3 = nn.Linear(16 * 5 * 5, 128) 
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 10)  

        # Create init layers
        self.init_layers()

    def forward(self, x):
        y = self.layer1(x)
        y = self.pool(nn.ReLU()(y))
        y = self.layer2(y)
        y = self.pool(nn.ReLU()(y))
        y = torch.flatten(y, 1)
        y = self.layer3(y)
        y = nn.ReLU()(y)
        y = self.layer4(y)
        y = nn.ReLU()(y) 
        output = self.layer5(y) # No need to add softmax as it will already being added in the loss function

        return output

    def init_layers(self):
        # Initialize the layers weights appropriately
        x = 23

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    ndi = 28*28
    ndo = 128

    
    fe = feature_extractor(ndi, ndo)
    classifier = classifier_model(ndi)
    print("Num params fe: ", count_parameters(fe))
    print("Num params clf: ", count_parameters(classifier))

    """
    x = torch.randn(16, 28*28)
    fe_out = fe(x)
    clf_out = classifier(x)
    """

    ndi = 32*32
    ndo = 128
    conv_fe = conv_feature_extractor(ndi, ndo)
    conv_classifier = conv_classifier_model(ndi)
    print("Num params conv fe: ", count_parameters(conv_fe))
    print("Num params conv clf: ", count_parameters(conv_classifier))

    x = torch.randn(3, 32, 32)
    x = x.reshape(-1, x.shape[0], x.shape[1], x.shape[2])
    fe_out = conv_fe(x)
    clf_out = conv_classifier(x)

    print("input shape: ", x.shape)
    print("conv fe output shape: ", fe_out.shape)
    print("clf_out shape: ", clf_out.shape)

if __name__ == "__main__":
    main()