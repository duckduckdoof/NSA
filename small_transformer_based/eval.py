import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from small_transformer_based.train import (
    CustomTokenizer,
    CombinedModel,
    evaluate_true,
)


import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from plots import return_task_grid
from llm.selector_prompt import generate_selector_prompt
import numpy as np

# Load the tokenizer and its vocabulary
tokenizer = CustomTokenizer()
tokenizer.load_vocab("vocab.json")  # Load the vocabulary

# Initialize the model with the correct vocab_size
model = CombinedModel(vocab_size=len(tokenizer.vocab))

# Conditional DataParallel wrapping
if torch.cuda.device_count() >= 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# Load the model checkpoint (adjust the path as needed)
print("##### LOADING THE MODEL... #####")
checkpoint = torch.load("small_transformer_based/results/best/checkpoint_epoch99_iter1.pth")

# Extract the model's state_dict
state_dict = checkpoint['model_state_dict']

# # Load the state_dict into the model
model.load_state_dict(state_dict)
print("#####... MODEL LOADED #####")

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


evaluate_true(model, tokenizer, "cuda", tta=False)
