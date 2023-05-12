import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline

from common import model
from kolihack.io import load_content_file


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


if __name__ == "__main__":
    df = load_content_file(truncate=True)
    df.describe()

    description_ = df.query("language == 'en'")['description']
    non_nan_descriptions = description_.dropna().reset_index(drop=True)

    non_nan_descriptions = non_nan_descriptions[0:1000]
    all_input_embeddings = encoder(non_nan_descriptions, model)

    df_i = pd.DataFrame(all_input_embeddings)
    df_i.to_pickle('input_embeddings.pkl')

    df_i = pd.DataFrame(non_nan_descriptions)
    df_i.to_pickle('non_nan_descriptions.pkl')
