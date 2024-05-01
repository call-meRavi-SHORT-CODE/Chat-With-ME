import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    

'''
1) NeuralNet class: 
This class inherits from nn.Module, which is the base class for all neural network modules in PyTorch.

2) Constructor (__init__):
It takes three parameters: input_size, hidden_size, and num_classes, which define the sizes of the input layer, hidden layer, and output layer (number of classes) respectively.

Inside the constructor, the layers of the neural network are defined:
self.l1: This is the first linear layer, which maps input features to hidden_size.
self.l2: This is the second linear layer, which maps hidden_size to hidden_size.
self.l3: This is the third linear layer, which maps hidden_size to num_classes (output size).
self.relu: This is the ReLU activation function, which is used after each linear layer to introduce non-linearity.

Forward method (forward):
This method defines the forward pass of the neural network.
It takes input x, which is passed through the layers defined in the constructor in sequence.
After each linear layer, the ReLU activation function is applied (self.relu) to introduce non-linearity.
The output of the final linear layer (self.l3) is returned without any activation function applied. This implies that the network output is the raw scores/logits, and no activation (like softmax) is applied to convert them into probabilities.'''