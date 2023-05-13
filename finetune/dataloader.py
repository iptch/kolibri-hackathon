from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class SearchDescriptionDataset(Dataset):
    def __init__(self, search_terms, descriptions, tokenizer, max_length=128):
        self.search_terms = search_terms
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.search_terms)

    def __getitem__(self, idx):
        search_term = self.search_terms[idx]
        description = self.descriptions[idx]

        # Encode the search term and description
        search_term_encoding = self.tokenizer.encode_plus(
            search_term,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        description_encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return search_term_encoding, description_encoding

# Instantiate the dataset and dataloader
dataset = SearchDescriptionDataset(search_terms, descriptions, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
