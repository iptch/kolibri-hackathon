import csv

def load_csv(filename):
    data=[]
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    for row in data:
        for column in ['correctlyspelled_search_terms', 'misspelled_search_terms', 'wrong_search_terms']:
            row[column] = row[column].split(',')

    assert(len(data) != 0)

    return data
    
def generate_samples(files):
    assert(len(files))
    search_terms = []
    content = []
    for f in files:
        datachunk = load_csv(f)
        for row in datachunk:
            number_to_duplicate_description = len(row['correctlyspelled_search_terms']) + len(row['misspelled_search_terms']) + len(row['wrong_search_terms'])
            #print(number_to_duplicate_description)           
            search_terms += row['correctlyspelled_search_terms']
            search_terms += row['misspelled_search_terms']
            search_terms += row['wrong_search_terms']
            text = 'Category: ' + row['category'] + '. Course Title: ' + row['title'] + '. Course Description:' + row['description']
            #print(text)
            for _ in range(number_to_duplicate_description):
                content.append(text)
            #print(len(content))

    assert(len(search_terms) == len(content))
    return search_terms, content