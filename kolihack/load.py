import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from kolihack.io import load_pkl_from_file
from kolihack.run import encoder

if __name__ == "__main__":
    input_embeddings = load_pkl_from_file("input_embeddings.pkl")
    non_nan_descriptions = load_pkl_from_file("non_nan_descriptions.pkl")
    print("Dear Kolibri User  - what are you looking for?")
    question = input()
    question_embedding = encoder([question], "bert-base-uncased")
    similarities_bert = cosine_similarity(input_embeddings, question_embedding)
    print(similarities_bert)

    index_of_highest_scores = np.argsort(similarities_bert, axis=0)[::-1][:3]
    results = non_nan_descriptions['description'][np.squeeze(index_of_highest_scores)]
    print(results)

