import argparse
from bert_search import BertSearch
from io import load_pkl_from_file

def main():
    parser = argparse.ArgumentParser(description="Query a search term, return the id of the content.")
    parser.add_argument("query", type=str, help="A search term you want to query")
    parser.add_argument("model", type=str, choices=["bert", "faiss"], help="select a model you want to query")
    args = parser.parse_args()

    assert(len(args.query) > 0)

if __name__ == "__main__":
    input_embeddings = load_pkl_from_file("../data/embeddings_by_id.pkl")
    input_id_and_text = load_pkl_from_file("../data/input_id_and_text.pkl")
    print("Dear Kolibri User  - what are you looking for?")

    if args.model == "bert":
        bert =BertSearch(input_embeddings, input_id_and_text)
        content_id = bert.search(args.query)
        print(content_id)
    else:
        assert(False)

# Call the main function
if __name__ == "__main__":
    main()


