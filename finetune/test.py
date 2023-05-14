import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


# Load the BERT model and tokenizer
model = BertEmbedding()
model.load_state_dict(torch.load('overfit_bert_base_uncased.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example input
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# Get embeddings
with torch.no_grad():
    embeddings = model(**inputs)

print(embeddings)