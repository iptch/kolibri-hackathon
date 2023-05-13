import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline

from common import model
from kolihack.io import load_content_file, load_pkl_from_file


def normalized_mean_pooling(token_vectors):
    sentences_vectors = [np.mean(tokens, axis=0) for tokens in token_vectors]
    normalized_embeddings = [vector / np.linalg.norm(vector) \
                             for vector in sentences_vectors]
    return normalized_embeddings


def encoder(texts, modelname):
    '''
    Use huggingface pipeline class to get vector embeddings for each token,
    then take the mean across tokens to get one vector embbedding per text
    '''
    pipe = pipeline("feature-extraction",
                    model=modelname,
                    tokenizer=modelname)

    embeddings = []
    loader = DataLoader(texts, batch_size=32, shuffle=False)
    for inputs in tqdm(loader):
        vectors = pipe(inputs)
        vectors = [np.vstack(item) for item in vectors]
        embs = normalized_mean_pooling(vectors)
        embeddings.extend(embs)
    return embeddings


def create_embeddings(input_id_and_text):
    all_input_embeddings = encoder(input_id_and_text['title'], model)
    df_i = pd.DataFrame(all_input_embeddings)
    df_i.to_pickle('all_input_embeddings_non_id.pkl')


if __name__ == "__main__":

    df = load_content_file(truncate=True)
    df.describe()
    input_id_and_text = df[['id', 'title']]

    # create_embeddings(input_id_and_text)

    input_embeddings = load_pkl_from_file("all_input_embeddings_non_id.pkl").values.tolist()

    embeddings_by_id = {}
    for index, row in input_id_and_text.iterrows():
        embeddings_by_id[row['id']] = input_embeddings[index]

    df_i = pd.DataFrame(embeddings_by_id)
    df_i.to_pickle('embeddings_by_id.pkl')
    df_i.to_csv('embeddings_by_id.csv')

    df_i = pd.DataFrame(input_id_and_text)
    df_i.to_pickle('input_id_and_text.pkl')
