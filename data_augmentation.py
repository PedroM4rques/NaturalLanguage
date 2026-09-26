import pandas as pd


train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')


def fix_nan_keywords(df: pd.DataFrame):
    keywords = set()

    for _, row in df.iterrows():
        if pd.notna(row['keywords']):
            keywords.update(row['keywords'].split(','))

        
    for idx, row in df.iterrows():

        row_kw = [kw for kw in row['keywords'].split(',') if kw] if pd.notna(row['keywords']) else []

        row_words = str(row['description']) + ' ' + str(row['sample_name']) + ' ' + str(row['transcription'])

        row_words = set(row_words.split(' '))

        common = list(keywords & row_words)

        row_kw.extend(kw for kw in common if kw and kw not in row_kw)

        df.at[idx, 'keywords'] = ','.join(row_kw)

    return df



train_df = fix_nan_keywords(train_df)
test_df = fix_nan_keywords(test_df)

train_df.to_csv('./data/train.csv', index=False)
test_df.to_csv('./data/test.csv', index=False)