import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common import model
from kolihack.io import load_pkl_from_file
from kolihack.run import encoder

if __name__ == "__main__":
    input_embeddings = load_pkl_from_file("embeddings_by_id.pkl")
    # input_embeddings = load_pkl_from_file("input_embeddings.pkl")
    input_id_and_text = load_pkl_from_file("input_id_and_text.pkl")
    print("Dear Kolibri User  - what are you looking for?")

    input_embeddings = input_embeddings.transpose()

    while True:
        question = input()
        if question == 'exit':
            break
        question_embedding = encoder([question], model)
        similarities_bert = cosine_similarity(input_embeddings, question_embedding)

        index_of_highest_scores = np.argsort(similarities_bert, axis=0)[::-1][:5]
        results = input_id_and_text.iloc[list(np.squeeze(index_of_highest_scores))]

        for _, row in results.iterrows():
            print(f"{row['id']:25}: {row['title']}")
