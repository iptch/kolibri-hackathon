import os
import openai
from utils import *
import argparse
import pandas as pd
import csv
import numpy as np
import sys
import signal

openai.api_key = os.environ.get("CHATGPT_API_KEY")

NUMBER_OF_SEARCHTERMS = 10

pkl_file = 'en_searchterms.pkl'

def askChatGPT(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=70,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    assert(response.choices)
    if response.choices:
         return response.choices[0].text

def generate_searchterms(title, description):
    prompt=f"Prompt:\nSimulate {NUMBER_OF_SEARCHTERMS} search text given the online course title and description. Title: {title}. Description: {description}. Write result in one line, seperated by comma. You must generate search terms using english."
    response = askChatGPT(prompt)
    lines = split_lines_and_remove_duplicate_lines(response)

    # if this happens, means chatgpt is returning a string in a wrong format
    if len(lines)!=1:
        print(f'chatgpt is not following the prompting format instruction on, generate list of {NUMBER_OF_SEARCHTERMS} empty search terms')
        searchterms = ['' for _ in range(NUMBER_OF_SEARCHTERMS)]

    # if this happens, means chatgpt is returning a string in a wrong format
    # append them to a fixed length
    searchterms = lines[0].split(',')
    if(len(searchterms) < NUMBER_OF_SEARCHTERMS):
        lengthBeforeAppend = len(searchterms)
        for i in range(lengthBeforeAppend, NUMBER_OF_SEARCHTERMS):
            searchterms.append('')
    else:
        searchterms = searchterms[0:10]
        

    for i in range(NUMBER_OF_SEARCHTERMS):
        searchterms[i] = remove_leading_trailing_whitespace(searchterms[i])

    print(searchterms)
    return searchterms

def main():
    parser = argparse.ArgumentParser(description="Generate search terms for the content.csv from Kaggle.")
    parser.add_argument("file", type=str, help="Location of content.csv")
    args = parser.parse_args()

    try:
        # Read the CSV file using pandas
        dataframe = pd.read_csv(args.file, nrows=20)

        en_searchterms = []
        column_names = [f'searchterm_{i}' for i in range(NUMBER_OF_SEARCHTERMS)]
        column_names.insert(0, 'id')
        column_names.insert(0, 'index')

        for index, row in dataframe.iterrows():
            language = row['language']
            if language == 'en':
                title = row['title']
                description = row['description']
                searchterms = generate_searchterms(title, description)
                searchterms.insert(0, row['id'])
                searchterms.insert(0,index)
                en_searchterms.append(searchterms)

        searchterm_dataframe = pd.DataFrame(en_searchterms, columns=column_names)
            
        # Save the updated DataFrame to a new CSV file
        searchterm_dataframe.to_pickle(pkl_file)

        print("updated dataframe is dumped into a new csv file.")

        # Function to handle keyboard interrupt signal
        def signal_handler(signal, searchterm_dataframe):
            print("Program interrupted. Saving searchterm_dataframe to a file...")
            searchterm_dataframe.to_pickle(pkl_file)
            print("searchterm_dataframe saved. Exiting the program.")
            sys.exit(0)

        # Assign the keyboard interrupt signal to the signal handler function
        signal.signal(signal.SIGINT, signal_handler)

    except FileNotFoundError:
        print("Error: The specified file does not exist.")

    except Exception as e:
        print("An error occurred:", str(e))

 
# Call the main function
if __name__ == "__main__":
    main()








