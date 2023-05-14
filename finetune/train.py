from BertEmbedding import BertEmbedding
import torch
from torch import nn
from torch.optim import Adam
from SearchDescriptionDataset import init_dataloader

device = torch.device('cpu')
num_epochs = 1

def train():
    
    dataloader = init_dataloader()

    # Load the BERT model and tokenizer
    model = BertEmbedding()

    # Define the loss (Here we use CosineEmbeddingLoss which is suitable for measuring distance between embeddings)
    criterion = nn.CosineEmbeddingLoss()

    # Define the optimizer
    optimizer = Adam(model.parameters())

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
            search_term_embedding = model(search_term.input_ids.squeeze(), search_term.attention_mask.squeeze())
            description_embedding = model(description.input_ids.squeeze(), description.attention_mask.squeeze())

            # cosine similarity loss expect the input1 and input2 to be 2D
            # But BERT model generate one embedding for each token, not for the whole text. So we get 3D embedding
            # We use the most simple way as a starting point, that is average the 3D embedging to get 2D.
            # We lose token-level embeddings in this way, but as a starting point could be a good choice
            search_term_embedding_averaged = search_term_embedding.mean(dim=1)
            description_embedding_averaged = description_embedding.mean(dim=1)

            # Compute the loss
            # Here we use a target of 1, indicating that search_term and description are a positive pair
            loss = criterion(search_term_embedding_averaged, description_embedding_averaged, torch.ones(search_term_embedding.size(0)).to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # You might want to print the loss and evaluate the model on a validation set here
        print(f'Epoch {epoch+1}, loss: {running_loss / num_batches}')

    torch.save(model.state_dict(), 'overfit_bert_base_uncased.pth')

if __name__ == "__main__":
    train()