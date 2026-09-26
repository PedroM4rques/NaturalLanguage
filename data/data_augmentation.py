from pathlib import Path
import pandas as pd


TRAIN_PATH = Path(__file__).parent / 'train.csv'
TEST_PATH  = Path(__file__).parent / 'test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)


def build_keywords(df1: pd.DataFrame, df2: pd.DataFrame) -> set:

    keywords = set()

    for df in (df1, df2):
        for _, row in df.iterrows():
            if pd.notna(row['keywords']):
                keywords.update(kw for kw in row['keywords'].split(',') if kw)

    return keywords


def augment_keywords(df: pd.DataFrame, keywords: set) -> pd.DataFrame:

    for idx, row in df.iterrows():

        row_kw = [kw for kw in row['keywords'].split(',') if kw] if pd.notna(row['keywords']) else []

        row_words = str(row['description']) + ' ' + str(row['sample_name']) + ' ' + str(row['transcription'])

        row_words = set(row_words.split(' '))

        common = list(keywords & row_words)

        row_kw.extend(kw for kw in common if kw and (kw not in row_kw))

        df.at[idx, 'keywords'] = ','.join(row_kw)

    return df



keywords = build_keywords(train_df, test_df)

train_df = augment_keywords(train_df, keywords)
test_df  = augment_keywords(test_df, keywords)


train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
