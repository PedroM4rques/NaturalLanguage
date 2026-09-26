import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


STOP_WORDS = ENGLISH_STOP_WORDS - {'back'}

train_df = pd.read_csv(config.TRAIN_PATH)
test_df  = pd.read_csv(config.TEST_PATH)


def is_valid_keyword(kw: str) -> bool:
    return (len(kw) > 1) and (not kw.isdigit()) and (kw not in STOP_WORDS)


def build_keywords(df1: pd.DataFrame, df2: pd.DataFrame) -> set:

    keywords = set()

    for df in (df1, df2):
        for _, row in df.iterrows():
            if pd.notna(row['keywords']):
                keywords.update(kw for kw in row['keywords'].split(',') if is_valid_keyword(kw))
    return keywords


def least_informative(df: pd.DataFrame, keywords: set, frac: float) -> set:

    row_words = df.apply(lambda row: list(set((str(row['description']) + ' ' + str(row['sample_name']) + ' ' + str(row['transcription'])).split(' ')) & keywords), axis=1)

    kw = pd.DataFrame({'keyword': row_words, 'medical_specialty': df['medical_specialty']}).explode('keyword').dropna().reset_index(drop=True)

    class_size = df['medical_specialty'].value_counts()

    # % de amostras de cada classe com a keyword -> distribuição pelas classes
    share = pd.crosstab(kw['keyword'], kw['medical_specialty']).div(class_size, axis=1)
    p     = share.div(share.sum(axis=1), axis=0)

    # 1 = espalhada igualmente por todas as classes (geral), 0 = só numa classe
    entropy = -(p * np.log(p.where(p > 0, 1))).sum(axis=1) / np.log(len(class_size))

    return set(entropy[entropy >= entropy.quantile(1 - frac)].index)


def augment_keywords(df: pd.DataFrame, keywords: set) -> pd.DataFrame:

    for idx, row in df.iterrows():

        if pd.notna(row['keywords']) and row['keywords'].strip(','):    # ONLY rows without keywords
            continue

        row_kw = []

        row_words = str(row['description']) + ' ' + str(row['sample_name']) + ' ' + str(row['transcription'])

        row_words = set(row_words.split(' '))

        common = list(keywords & row_words)

        row_kw.extend(kw for kw in common if kw not in row_kw)

        df.at[idx, 'keywords'] = ','.join(row_kw)

    return df



keywords = build_keywords(train_df, test_df)
keywords = keywords - least_informative(train_df, keywords, config.DROP_FRAC)    # só train tem labels

train_df = augment_keywords(train_df, keywords)
test_df  = augment_keywords(test_df, keywords)


train_df.to_csv(config.TRAIN_PATH, index=False)
test_df.to_csv(config.TEST_PATH, index=False)
