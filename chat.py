import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Ravi"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Sorry I could not understant"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)



"""
1) You load the intents from a JSON file named intents.json into a Python dictionary.

2)You load the pre-trained model and associated data (input size, hidden size, output size, etc.) from a file named data.pth.

3) You define the get_response function, which takes a user's message as input. Inside this function:
 -The user's message is tokenized and converted into a bag-of-words representation using the tokenize and bag_of_words functions from nltk_utils.
 -The bag-of-words representation is converted into a PyTorch tensor and moved to the appropriate device (CPU or GPU).
 -The model processes the input tensor and predicts the most likely intent tag for the user's message.

4)If the predicted intent tag has a probability above a certain threshold (0.75 in this case), a response is randomly chosen from the corresponding intent's responses in the intents dictionary.


5)If the confidence is below the threshold or if no matching intent is found, a default response indicating inability to answer is returned.

Finally, in the __main__ block:
The code prompts the user to input a message.
The get_response function is called with the user's input.
If the input is "quit", the loop terminates, otherwise, the response is printed."""