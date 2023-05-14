from SearchDescriptionDataset import init_dataloader
from torch.nn.functional import cosine_similarity
from BertEmbedding import BertEmbedding
import torch
from torch import nn
# Load the BERT model and tokenizer
model = BertEmbedding()
#model.load_state_dict(torch.load('overfit_bert_base_uncased.pth'))
model.eval()
torch.no_grad()

def inference():
    dataloader = init_dataloader(batch_size=1)

    # Load the BERT model and tokenizer
    model = BertEmbedding()

    # Define the loss (Here we use CosineEmbeddingLoss which is suitable for measuring distance between embeddings)
    criterion = nn.CosineEmbeddingLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for batch in dataloader:
        search_term, description = batch
        search_term = search_term.to(device)
        description = description.to(device)

        # Forward pass for search term and description
        search_term_embedding = model(search_term.input_ids.squeeze(1), search_term.attention_mask.squeeze(1))
        description_embedding = model(description.input_ids.squeeze(1), description.attention_mask.squeeze(1))

        # cosine similarity loss expect the input1 and input2 to be 2D
        # But BERT model generate one embedding for each token, not for the whole text. So we get 3D embedding
        # We use the most simple way as a starting point, that is average the 3D embedging to get 2D.
        # We lose token-level embeddings in this way, but as a starting point could be a good choice
        search_term_embedding_averaged = search_term_embedding.mean(dim=1)
        description_embedding_averaged = description_embedding.mean(dim=1)

        # size should be [1]
        similarity = cosine_similarity(search_term_embedding_averaged, description_embedding_averaged, dim=1)

        print(similarity)

if __name__ == "__main__":
    inference()