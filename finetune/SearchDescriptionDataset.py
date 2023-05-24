from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from utils import generate_samples

def init_dataloader(batch_size = 1):
    # Instantiate the dataset and dataloader
    csvfiles = ['English_1.csv','English_2.csv', 'English_3.csv', 'English_4.csv', 'English_5.csv', 'English_6.csv']
    for i in range(len(csvfiles)):
        csvfiles[i]= 'generateDataFromChatGPT/' + csvfiles[i]
    print(csvfiles)

    search_terms, content = generate_samples(csvfiles)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #TODO: try the finetuned tokenizer?

    dataset = SearchDescriptionDataset(search_terms, content, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_size = len(dataset)
    batch_size = dataloader.batch_size
    num_batches = dataset_size // batch_size
    print('\n\n\n\ndataset size', dataset_size, 'batch size', batch_size, 'number of batches', num_batches)
    return dataloader, dataset_size, batch_size, num_batches

class SearchDescriptionDataset(Dataset):
    def __init__(self, search_terms, descriptions, tokenizer, max_length=128):
        assert(len(search_terms) == len(descriptions))
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

if __name__ == "__main__":
    csvfiles = ['English_1.csv','English_2.csv', 'English_3.csv', 'English_4.csv', 'English_5.csv', 'English_6.csv']
    for i in range(len(csvfiles)):
        csvfiles[i]= 'generateDataFromChatGPT/' + csvfiles[i]
    print(csvfiles)

    search_terms, content = generate_samples(csvfiles)
    print(f'In total we have \n\t{len(search_terms)} search terms and {len(content)} description generated')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = SearchDescriptionDataset(search_terms, content, tokenizer)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    dataset_size = len(dataset)
    batch_size = dataloader.batch_size
    num_batches = dataset_size // batch_size

    for bi, batch in enumerate(dataloader):
        search_term_token, content_token = batch
        if bi == 5:
            print('search term token', search_term_token)
            print('content token', content_token) 

    
    print('\n\n\n\ndataset size', dataset_size, 'batch size', batch_size, 'number of batches', num_batches)
