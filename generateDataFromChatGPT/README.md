# Generate Syntetic Data
* Prerequiests: apply an openai API key
* Dependencies: openapi, argparse
* Run: `python3 generateSyntheticData.py English 10`
* Output: `{language}_x.csv` with 10 rows, 6 columns (title, description, category, correct spelled search terms, misspelled search terms, wrong search terms)
* Location: The csv files will dump to the directory, where you run the command.
* Constraints: Only English, Brazilian, Portugues and Spanish are supported. Suggested is to run per time with only 10 samples.

# Generate search terms for the content.csv from Kaggle
You probablly already have partial data just like the content.csv, but you only have title and descriprtion as the content. But you have never stored search terms for these content before, and you want to use chatgpt to simulate some search terms for you.
Once you generate search terms for your content, you can use it to overfit your model.
