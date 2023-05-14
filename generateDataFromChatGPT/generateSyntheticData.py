import os
import openai
from utils import *
import argparse
from argparse import RawTextHelpFormatter

openai.api_key = os.environ.get("CHATGPT_API_KEY")

def askChatGPT(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )
    assert(response.choices)
    if response.choices:
         return response.choices[0].text

def generate_course_data(language, number):
    course_data = []
    for _ in range(number):
        prompt=f"Prompt:\nGenerate a title, a description, a category, 5 correctly spelled search terms, 5 misspelled search terms, 5 wrong search terms in {language} for an online video using the following format:\n\nTitle: [Generate a title for the online course video]\nDescription: [Generate a description for the online course video]\nCategory: [Generate a category for the online course video, category must write in lowercase letters, must do not contain leading or trailing empty space, must be a single category. Also, the category must be one of these: mathematics, language, chemistry, biology, algorithms, deep learning, computer vision, programming languages, software architecture, frontend development, backend development.]\nCorrectly spelled search terms: [Generate 5 possible correctly spelled serach terms, seperate them using comma]\nMisspelled search terms: [Generate 5 possible misspelled serach terms, seperate them using comma]\nWrong search terms: [Generate 5 wrong search terms, seperate them using comma]"
        response = askChatGPT(prompt)
        lines = split_lines_and_remove_duplicate_lines(response)

        # strong constraints, ignore response directly
        if len(lines)!=6:
           print(response)
           continue

        # format        
        title = lines[0].split('Title: ')[-1]
        title = remove_leading_trailing_whitespace(title)
        description = lines[1].split('Description: ')[-1]
        description = remove_leading_trailing_whitespace(description)
        category= lines[2].split('Category: ')[-1]
        category = remove_leading_trailing_whitespace(category)
        correctly_spelled_search_terms = lines[3].split('Correctly spelled search terms: ')[-1]
        correctly_spelled_search_terms = remove_leading_trailing_whitespace(correctly_spelled_search_terms)
        misspelled_search_terms= lines[4].split('Misspelled search terms: ')[-1]
        misspelled_search_terms = remove_leading_trailing_whitespace(misspelled_search_terms)
        wrong_search_terms = lines[5].split('Wrong search terms: ')[-1]
        wrong_search_terms = remove_leading_trailing_whitespace(wrong_search_terms)

        # strong constraints, ignore response directly
        if len(title) == 0 or len(description) == 0 or len(category) == 0:
            continue

        # weak constraints, only print to screen for monitoring
        if category not in category_labels:
            print("monitoring:", category)
        if len(correctly_spelled_search_terms.split(',')) !=5:
            print("monitoring:", correctly_spelled_search_terms)
        if len(misspelled_search_terms.split(',')) !=5:
            print("monitoring:", misspelled_search_terms)
        if len(wrong_search_terms.split(',')) !=5:
            print("monitoring:", wrong_search_terms)

        course_data.append( {
            'title': title,
            'description': description, 
            'category': category, 
            'correctlyspelled_search_terms': correctly_spelled_search_terms, 
            'misspelled_search_terms': misspelled_search_terms,
            'wrong_search_terms': wrong_search_terms
        })

    assert(len(course_data) > 0)
    return course_data

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, description="Generate CSV files with incremented indices. Be careful you could overwrite the file.\n\n\
        Example to use: python3 generateSyntheticData.py English 10\n\n\
        With this command you are going to generate 10 content-search terms pairs, the content is title and description.\n\n\
        The search terms include:\n\
        1. Correct spelled search terms, you can use them to create positve samples\n\
        2. Misspelled search terms: you can use them to create positie samples, but your model will generalize better on misspelling\n\
        3. Wrong search terms: You can use them to create negative samples, so that your model knows better on what is not correct\n")
    parser.add_argument("language", type=str, choices=["English","Spanish","Brazilian","Portugues"], help="Possible languages: English, Spanish, Brazilian, Portugues")
    parser.add_argument("number", type=max_samples_number_per_run, help="Number of samples you want to generate,  1 - 100")
    args = parser.parse_args()

    selectedLanguage = capitalize_word(args.language)
    file_path = generate_csv_file(selectedLanguage)
    course_data = generate_course_data(selectedLanguage, args.number)
    dump_csv(course_data, file_path)

def max_samples_number_per_run(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 100:
        raise argparse.ArgumentTypeError("Value must be between 1 and 100")
    return ivalue

# Call the main function
if __name__ == "__main__":
    main()








