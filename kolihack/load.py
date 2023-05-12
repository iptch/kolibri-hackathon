import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common import model
from kolihack.io import load_pkl_from_file
from kolihack.run import encoder

if __name__ == "__main__":
    input_embeddings = load_pkl_from_file("input_embeddings.pkl")
    non_nan_descriptions = load_pkl_from_file("non_nan_descriptions.pkl")
    print("Dear Kolibri User  - what are you looking for?")
    question = input()
    question_embedding = encoder([question], model)
    similarities_bert = cosine_similarity(input_embeddings, question_embedding)

    index_of_highest_scores = np.argsort(similarities_bert, axis=0)[::-1][:10]
    results = non_nan_descriptions['description'][np.squeeze(index_of_highest_scores)]
    for result in results:
        print(result)
        print("---------")
