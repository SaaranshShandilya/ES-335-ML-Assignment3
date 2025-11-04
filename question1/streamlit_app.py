import streamlit_app as st
import pickle
import numpy as np
import torch
from torch import nn
from torch.functional import F

st.title('Next word predictor')
st.write('Enter the context:')
context = st.text_input("Enter the context here")

options = ['Greedy', 'Sampling']
st.sidebar.title("Settings")
block_size = st.sidebar.slider("Block size", min_value=0, max_value = 5)
approach = st.sidebar.selectbox(
    "Output generation approach",
    options
)

temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value = 2.0)
embedding_dimension = st.sidebar.selectbox(
    "Input embedding dimension",
    [32, 64]
)
max_k_words = st.sidebar.slider("Max k words", min_value = 5, max_value = 20)
random_seed = st.sidebar.text_input("Random seed", value = 42)
activation_function = st.sidebar.selectbox(
    "Activation function",
    [F.relu, F.tanh]
)

class NextWord(nn.Module):
    def __init__(self, vocab_size, context_size=5, embed_dim=embedding_dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        input_dim = context_size * embed_dim
        self.hidden_layer1 = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.4)
        self.hidden_layer2 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(512, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.shape[0], -1)
        h2 = activation_function(self.hidden_layer1(embeds))
        h2 = self.dropout(h2)
        h2=activation_function(self.hidden_layer2(h2))
        h2 = self.dropout(h2)
        logits = activation_function(self.output(h2))
        return logits

model = torch.load('natural_language.pth', weights_only=False)

with open('exported_variables.pkl', 'rb') as f:
    stoi = pickle.load(f)
    itos = pickle.load(f)




model = model.to("cuda")
g = torch.Generator()
g.manual_seed(int(random_seed))
def generate_name(model, itos, stoi,existing_string,approach, block_size=5, max_len=20,temperature = 1.0):
    model.eval()
    context = []
    existing_string = existing_string.lower()
    if(len(existing_string)>0):
        wr = existing_string.split(" ")
        if(len(wr)>block_size):
            wr=wr[-block_size:]
        else:
            while len(wr) < block_size:
                wr.insert(0,'_')
        for ele in wr:
            context.append(stoi[ele])
    else:
        context = [0] * block_size
    while len(context) < 5:
        context.insert(0,0)
    with torch.no_grad():
        sentence = ''
        for i in range(max_len):
            x = torch.tensor(context).view(1,-1).to("cuda")
            y_pred = model(x)
            y_pred = y_pred / temperature
            if(approach == "Greedy"):
                probs = torch.softmax(y_pred, dim=-1)
                ix = torch.argmax(probs, dim=-1).item()
            else:
                ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
                word = itos[ix]
            word = itos[ix]
            if word == '.':
                sentence+='.'
                break
            sentence+=" "+ word
            context = context[1:] + [ix]
            i=i+1
    model.train()
    return sentence



if st.button("Predict"):
    sentence = generate_name(model, itos, stoi, context,approach, block_size,max_k_words, temperature )
    st.write(f"Complete sentence: {context+sentence}")
