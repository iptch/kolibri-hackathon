import re
import time
from os import listdir
from os.path import isfile, join
from statistics import mean

import faiss
import pandas as pd
from matplotlib import pyplot as plt

from kolihack.bert_search import BertSearch
from kolihack.io import load_pkl_from_file

METRIC_MEMORY_USAGE = 'memory_usage'
METRIC_CPU_TIME = 'cpu_time'
METRIC_EXEC_TIME = 'exec_time'

import tracemalloc


class SearchResults:
    def __init__(self, query, search_results, execution_time, cpu_time, memory_usage):
        self._query = query
        self._search_results = search_results
        self._execution_time = execution_time
        self._cpu_time = cpu_time
        self._memory_usage = memory_usage

    @property
    def query(self):
        return self._query

    @property
    def search_results(self):
        return self._search_results

    @property
    def execution_time(self):
        return self._execution_time

    @property
    def cpu_time(self):
        return self._cpu_time

    @property
    def memory_usage(self):
        return self._memory_usage


def get_title_from_ids(ids, ids_texts):
    return [ids_texts.loc[ids_texts['id'] == id]['title'].values[0] for id in ids]


def test_search(query, search_object, ids_text):
    tracemalloc.start()
    t_st = time.time()
    t_p_st = time.process_time()
    results = search_object.search(query)
    memory_usage = tracemalloc.get_traced_memory()
    t_end = time.time()
    t_p_end = time.process_time()
    tracemalloc.stop()
    search_result = SearchResults(query, get_title_from_ids(results, ids_text), t_end - t_st, t_p_end - t_p_st,
                                  memory_usage[1])
    try:
        for _, row in results.iterrows():
            print(f"{row['id']:25}: {row['title']}")
    except:
        for id in results:
            try:
                print(ids_text.loc[ids_text['id'] == id]['title'].values[0])
            except:
                print(id)
    return search_result


def load_embeddings():
    return load_pkl_from_file("embeddings_by_id.pkl").transpose(), load_pkl_from_file("input_id_and_text.pkl")


def load_faiss_index():
    return faiss.read_index("faiss_index.bin")


def process_queries(queries, search_object, ids_text):
    specs = []
    for query in queries:
        specs.append(test_search(query, search_object, ids_text))
    plane_pd = pd.DataFrame([[result.query, result.search_results[0], result.search_results[1],
                              result.search_results[2], result.execution_time, result.cpu_time, result.memory_usage] for
                             result in specs],
                            columns=['Query', 'Result1', 'Result2', 'Result3', 'exec_time', 'cpu_time', 'memory_usage'])
    print(plane_pd)
    plane_pd.to_pickle('results/all-mpnet-base-v2_results.pkl')


def plot():
    mypath = "results/"
    all_perf_test_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    all_results = {}

    for file in all_perf_test_files:
        model_name = re.search(r"^(.*?)_results\.pkl", file).group(1)
        data = load_pkl_from_file(f"{mypath}{file}")

        all_results[model_name] = {
            METRIC_EXEC_TIME: mean(data[METRIC_EXEC_TIME]),
            METRIC_CPU_TIME: mean(data[METRIC_CPU_TIME]),
            METRIC_MEMORY_USAGE: mean(data[METRIC_MEMORY_USAGE])}

        all_model_names = all_results.keys()

        plot_metric(all_model_names, all_results, METRIC_CPU_TIME)
        plot_metric(all_model_names, all_results, METRIC_EXEC_TIME)
        plot_metric(all_model_names, all_results, METRIC_MEMORY_USAGE)


def plot_metric(all_model_names, all_results, metric='cpu_time'):
    cpu_times = [v[metric] for k, v in all_results.items()]
    fig, ax = plt.subplots()
    bar_labels = all_model_names
    bar_colors = ['tab:red', 'tab:blue']
    ax.bar(all_model_names, cpu_times, label=bar_labels, color=bar_colors)
    ax.set_ylabel(metric)
    ax.set_title(f'Comparison by model by {metric}')
    ax.legend(title='Model')
    plt.show()


if __name__ == "__main__":
    embeddings_ids, ids_text = load_embeddings()
    #faiss_index = load_faiss_index()
    bert_search = BertSearch(embeddings_ids, ids_text)
    #bert_basic_search = BertSearch(embeddings_ids, ids_text,  "bert-base-uncased")
    #faiss_search = FaissSearch(embeddings_ids, faiss_index)
    #bert_MIniLM_search = BertSearch(embeddings, "sentence-transformers/all-MiniLM-L6-v2")
    queries = ['bird', 'i want to learn something about biology', 'what is pythagoras',
               'find the value y dependent on x', 'moon and the sun', 'how to grow crops', 'fish', 'healthy food',
               'when to plant seeds']
    process_queries(queries, bert_search, ids_text)

    plot()
"""    while True:
        print("Dear kolibiri user - what are you looking for?")
        query = input()
        if query == 'exit': break

        search_result = test_search(query, bert_search)
        #search_result = test_search(query, bert_basic_search)
        search_restul = test_search(query, faiss_search)
        #search_result = test_search(query, bert_MIniLM_search, non_nan_descriptions)"""
