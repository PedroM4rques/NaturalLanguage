import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

import config


def split_keywords(keywords: str) -> list[str]:
    return [kw for kw in keywords.split(',') if kw]


def build_model():

    # onehot
    vectorizer = CountVectorizer(tokenizer=split_keywords, token_pattern=None, lowercase=False, binary=True)
    clf        = LogisticRegression(C=config.LR_C, class_weight='balanced', max_iter=3000)

    return make_pipeline(vectorizer, clf)


def main():

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df  = pd.read_csv(config.TEST_PATH)

    X_train = train_df['keywords'].fillna('')
    X_test  = test_df['keywords'].fillna('')
    y       = train_df['medical_specialty']
    classes = sorted(y.unique())


    # ---- k-fold
    cv   = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    pred = cross_val_predict(build_model(), X_train, y, cv=cv)

    print(f"Logistic regression, keywords one-hot (C = {config.LR_C}) | {config.CV_FOLDS}-fold CV\n")
    print(f"Accuracy:         {accuracy_score(y, pred):.4f}")
    print(f"Macro F1:         {f1_score(y, pred, average='macro'):.4f}")

    f1 = pd.Series(f1_score(y, pred, average=None, labels=classes), index=classes)
    print("\nPer-class F1:")
    print(f1.sort_values().round(4).to_string())

    below = f1[f1 < 0.25]
    print(f"\n{'FAIL' if len(below) else 'OK'}: {len(below)} class(es) with F1 below 25%{': ' + ', '.join(below.index) if len(below) else ''}")


    # results.txt
    model = build_model().fit(X_train, y)
    print(f"\nVocabulary: {len(model[0].vocabulary_)} keywords")

    # linhas vazias
    most_common = y.mode()[0]
    no_features = model[0].transform(X_test).getnnz(axis=1) == 0

    test_pred = pd.Series(model.predict(X_test)).mask(no_features, most_common)
    config.restore_label_case(test_pred).to_csv(config.RESULTS_PATH, index=False, header=False)

    print(f"Test: {len(test_pred)} predictions -> {config.RESULTS_PATH.name} ({no_features.sum()} without features -> '{most_common}')")



if __name__ == '__main__': main()
