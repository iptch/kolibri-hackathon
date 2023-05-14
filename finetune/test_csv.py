import csv

filename = 'generateDataFromChatGPT/English_0.csv'

with open(filename, 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Convert the string representation of lists into actual lists
for row in data:
    print('row type', type(row))
    
    print(row['title'])
    print(row['description'])
    print(row['category'])
    for column in ['correctlyspelled_search_terms', 'misspelled_search_terms', 'wrong_search_terms']:
        row[column] = row[column].split(',')
        print(len(row[column]))
        #print(row[column])

print(type(data))
