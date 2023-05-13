import numpy as np
import time
from kolihack.io import load_pkl_from_file
from kolihack.bert_search import BertSearch


class SearchResults:
    def __init__(self, search_results, execution_time, cpu_time, memory_usage):
        self.search_results = search_results
        self.execution_time = execution_time
        self.cpu_time = cpu_time
        self.memory_usage = memory_usage


def test_search(query, search_object, descriptions):
    exex_time_start = time.time()
    cpu_time_start = time.process_time()
    indexes_highest_ranking = search_object.search(query)
    results = descriptions['description'][np.squeeze(indexes_highest_ranking)]
    exex_time_end = time.time()
    cpu_time_end = time.process_time()
    search_result = SearchResults(results, exex_time_end-exex_time_start, cpu_time_end-cpu_time_start,0 )
    print(search_result.search_results)
    print(f'Execution Time {search_result.execution_time} \n CpuTime {search_result.execution_time} \n ')



def load_embeddings():
    return load_pkl_from_file("input_embeddings.pkl"), load_pkl_from_file("non_nan_descriptions.pkl")


if __name__ == "__main__":
    embeddings, non_nan_descriptions = load_embeddings()
    bert_search = BertSearch(embeddings)
    while True:
        print("Dear kolibiri user - what are you looking for?")
        query = input()
        if query == 'exit': break

        st = time.time()
        search_result = test_search(query, bert_search, non_nan_descriptions)
        ed = time.time()
        print(f'time: {ed-st}')
