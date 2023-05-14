import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss

from kolihack.io import load_content_file, load_pkl_from_file


class FaissSearch:
    def __init__(self, embeddings_ids="", faiss_index="", model_name="msmarco-MiniLM-L-6-v3", n_results=5):
        self.embeddings_ids = embeddings_ids
        self.faiss_index = faiss_index
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.n_results = n_results

    def search(self, query):
        query_vector = self.model.encode([query])
        top_k = self.faiss_index.search(query_vector, self.n_results)
        return [self.embeddings_ids.index[_id] for _id in top_k[1]][0]

    def saveIndex(self):
        data_list = load_pkl_from_file("input_id_and_text.pkl")['title']

        # embed reviews
        description_embeds = self.model.encode(data_list, show_progress_bar=True)

        description_embeds = np.array([embedding for embedding in description_embeds]).astype("float32")
        index = faiss.IndexFlatL2(description_embeds.shape[1])
        index.add(description_embeds)

        faiss.write_index(index, "faiss_index.bin")


if __name__ == "__main__":
    ff = FaissSearch()
    ff.saveIndex()
