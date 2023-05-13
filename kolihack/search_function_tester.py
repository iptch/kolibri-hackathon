import numpy as np
import time
from kolihack.io import load_pkl_from_file
from kolihack.bert_search import BertSearch
from functools import wraps
from time import time

def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

class SearchResults:
    def __init__(self, search_results, execution_time, cpu_time, memory_usage):
        self.search_results = search_results
        self.execution_time = execution_time
        self.cpu_time = cpu_time
        self.memory_usage = memory_usage


@timeit
def test_search(query, search_object):
    results = search_object.search(query)
    search_result = SearchResults(results, 2, 3,0 )
    for _, row in results.iterrows():
        print(f"{row['id']:25}: {row['title']}")



def load_embeddings():
    return load_pkl_from_file("embeddings_by_id.pkl").transpose(), load_pkl_from_file("input_id_and_text.pkl")


if __name__ == "__main__":
    embeddings_ids, ids_text = load_embeddings()
    bert_search = BertSearch(embeddings_ids, ids_text)
    bert_basic_search = BertSearch(embeddings_ids, ids_text,  "bert-base-uncased")
    #bert_MIniLM_search = BertSearch(embeddings, "sentence-transformers/all-MiniLM-L6-v2")
    while True:
        print("Dear kolibiri user - what are you looking for?")
        query = input()
        if query == 'exit': break

        search_result = test_search(query, bert_search)
        search_result = test_search(query, bert_basic_search)
        #search_result = test_search(query, bert_MIniLM_search, non_nan_descriptions)


