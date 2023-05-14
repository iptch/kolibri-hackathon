from dataloader import SearchDescriptionDataset
from SearchTermDescriptionModel import SearchTermDescriptionModel
from transformers import BertTokenizer
from torch import nn
from torch.optim import Adam

# Load the BERT model and tokenizer
model = SearchTermDescriptionModel()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Instantiate the dataset and dataloader
dataset = SearchDescriptionDataset(search_terms, descriptions, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the loss (Here we use CosineEmbeddingLoss which is suitable for measuring distance between embeddings)
criterion = nn.CosineEmbeddingLoss()

# Define the optimizer
optimizer = Adam(model.parameters())

def train():
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    # Assume that dataloader is an instance of torch.utils.data.DataLoader
    # and provides batches of (search_term, description) pairs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            # Zero out gradients from the last run
            optimizer.zero_grad()

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

            running_loss += loss.item()
        
        # You might want to print the loss and evaluate the model on a validation set here
        print(f'Epoch {epoch+1}, loss: {running_loss / num_batches}')

    torch.save(model.state_dict(), 'overfit_bert_base_uncased.pth')