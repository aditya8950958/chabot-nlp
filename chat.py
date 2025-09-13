import random
import json
import torch
import numpy as np
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

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    
    # Convert X to a numpy array and reshape it
    X = np.array(X)
    X = X.reshape(1, X.shape[0])
    
    # Convert to a torch tensor and make sure it has the correct dtype (float32)
    X = torch.from_numpy(X).to(device).float()  # Convert to float32

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

if __name__ == "__main__":
    print("Chatbot is ready! (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input == "quit":
            break

        response = get_response(user_input)
        print(f"{bot_name}: {response}")