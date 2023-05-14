import os
import csv


category_labels = ['mathematics', 'language', 'chemistry', 'biology', 'algorithms', 'deep learning', 'computer vision', 'programming languages', 'software architecture', 'frontend development', 'backend development']

def get_largest_index(language):
    assert(len(language)!=0)
    largest_index = -1
    files = os.listdir('.')
    for file_name in files:
        if file_name.startswith(f'{language}_') and file_name.endswith('.csv'):
            try:
                index = int(file_name.split('_')[1].split('.')[0])
                largest_index = max(largest_index, index)
            except (ValueError, IndexError):
                continue
    return largest_index

def generate_csv_file(language):
    assert(len(language)!=0)
    largest_index = get_largest_index(language)
    new_index = largest_index + 1
    file_name = f'{language}_{new_index}.csv'
    return file_name

def remove_leading_trailing_whitespace(text):
    cleaned_text = text.strip()
    return cleaned_text

def split_lines_and_remove_duplicate_lines(text):
    # Split the text into lines
    lines = text.splitlines()

    # Remove duplicate newlines
    cleaned_lines = []
    for line in lines:
        if line.strip():  # Exclude empty lines
            cleaned_lines.append(line)

    return cleaned_lines

def capitalize_word(word):
    if word:
        return word[0].upper() + word[1:].lower()
    else:
        return word


def dump_csv(course_data, file_path):
    assert(len(course_data) > 0)

    # Extracting field names from the first dictionary
    field_names = course_data[0].keys()
    assert(len(field_names) !=0)

    # Writing data to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

        for row in course_data:
            writer.writerow(row)

    print('Data has been written to', file_path)