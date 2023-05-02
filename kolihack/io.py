import pandas as pd

from common import DATA_DIR_PATH


def load_content_file() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_DIR_PATH / 'content.csv')
        print(f'Successfully loaded pandas data frame from {DATA_DIR_PATH}/content.csv')
        return df
    except FileNotFoundError as error:
        print(f'Could not load content file. Did you download it to "data/content.csv"? {error}')
