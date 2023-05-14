import torch
from transformers import BertModel, BertTokenizer
from torch import nn
from torch.optim import Adam

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define the model
class SearchTermDescriptionModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output  # We use the [CLS] token embedding

model = SearchTermDescriptionModel(bert_model)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss (Here we use CosineEmbeddingLoss which is suitable for measuring distance between embeddings)
criterion = nn.CosineEmbeddingLoss()

# Define the optimizer
optimizer = Adam(model.parameters())

# Assume that dataloader is an instance of torch.utils.data.DataLoader
# and provides batches of (search_term, description) pairs
for epoch in range(num_epochs):
    for batch in dataloader:
        search_term, description = batch
        search_term = search_term.to(device)
        description = description.to(device)

        # Forward pass for search term and description
        search_term_embedding = model(*search_term)
        description_embedding = model(*description)

        # Compute the loss
        # Here we use a target of 1, indicating that search_term and description are a positive pair
        loss = criterion(search_term_embedding, description_embedding, torch.ones(search_term_embedding.size(0)).to(device))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # You might want to print the loss and evaluate the model on a validation set here
