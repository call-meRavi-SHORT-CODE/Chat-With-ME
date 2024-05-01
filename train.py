import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np

import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json','r') as f:
    intents = json.load(f)

all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
ignore_words= ['?','!',".",","]
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))

tags= sorted(set(tags))


x_train=[]
y_train=[]

for (pattern_sentence , tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)

    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)



x_train = np.array(x_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)



class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
"""
1) Constructor (__init__):
    It initializes the dataset with the training data.
        - self.n_samples: This stores the number of samples in the dataset, which is determined by the length of x_train.
        -self.x_data: This stores the input features (training data), presumably x_train.
        -self.y_data: This stores the labels (target data), presumably y_train.
2) __getitem__ method:
    This method allows the dataset to be indexed like dataset[i].
    It takes an index as input and returns the input features (x_data) and corresponding labels (y_data) for the sample at that index.
3) __len__ method:
    This method returns the length of the dataset, which is the number of samples (n_samples)."""



dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')



"""
to train a neural network model

1) Dataset and DataLoader Setup:
    It seems like you have a custom dataset (ChatDataset) and a DataLoader for loading batches of data during training.
    The DataLoader shuffles the data (shuffle=True) and specifies the number of workers for loading the data (num_workers).

2) Device Selection:
    This code checks whether a GPU (cuda) is available and assigns the device accordingly.
    If a GPU is available, it uses CUDA for training; otherwise, it falls back to CPU.

3) Model Initialization: 
    You're initializing a neural network model (NeuralNet) with specified input size, hidden size, and output size. 
    The model is then moved to the selected device.

4) Loss Function and Optimizer: 
    CrossEntropyLoss is chosen as the loss function, 
    which is suitable for multi-class classification tasks like this one. 
    Adam optimizer is used for optimizing the model parameters.

5) Training Loop: 
    The training loop iterates over each epoch and each batch of data. 
    It moves the input data and labels to the selected device. 
    Then, it performs a forward pass through the model, calculates the loss, performs backpropagation, 
    and updates the model parameters using the optimizer.

6) Logging: 
    Every 100 epochs, the code prints out the current epoch and the loss.
7) Final Loss: 
    After training completes, the final loss is printed out"""