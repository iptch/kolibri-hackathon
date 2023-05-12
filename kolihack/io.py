import pandas as pd

from common import DATA_DIR_PATH


def auto_truncate(val):
    return val[:511]


def load_content_file(path=DATA_DIR_PATH / 'content.csv', truncate=False) -> pd.DataFrame:
    try:
        if truncate:
            df = pd.read_csv(path, converters={'description': auto_truncate})
        else:
            df = pd.read_csv(path)
        print(f'Successfully loaded pandas data frame from {path}/content.csv')
        return df
    except FileNotFoundError as error:
        print(f'Could not load content file. Did you download it to "data/content.csv"? {error}')


def load_pkl_from_file(path) -> pd.DataFrame:
    try:
        df = pd.read_pickle(path)
        print(f'Successfully loaded pandas data frame from {path}')
        return df
    except FileNotFoundError as error:
        print(f'Could not load content file. Did you download it to "data/content.csv"? {error}')
