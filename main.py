from functools import partial

import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score, f1_score

import config


def split_keywords(keywords: str) -> list[str]:
    return [kw for kw in keywords.split(',') if kw]


def matched_pair_positions(tokens: list[str], classes: list[str]) -> list[int]:

    # cada par de keywords seguidas ('cardiovascular,pulmonary' -> 'cardiovascularpulmonary')
    # minimum edit distance (Levenshtein: ins/del/sub = 1), normalizada: 1 - dist / max(len(a), len(b))
    # só fica o par se a concatenação (quase) coincidir com o nome de uma classe -> devolve a posição da 1ª keyword do par
    positions = []
    for i, (a, b) in enumerate(zip(tokens, tokens[1:])):
        _, score, _ = process.extractOne(a + b, classes, scorer=Levenshtein.normalized_similarity)
        if score > config.MED_THRESHOLD:
            positions.append(i)

    return positions


def class_pairs(keywords: str, classes: list[str]) -> list[str]:
    tokens = split_keywords(keywords)
    return [f'{tokens[i]}|{tokens[i + 1]}' for i in matched_pair_positions(tokens, classes)]


def unpaired_keywords(keywords: str, classes: list[str]) -> list[str]:
    # keywords sozinhas, sem as que já foram concatenadas num par ('cardiovascular' e 'pulmonary' só contam como par)
    tokens = split_keywords(keywords)
    paired = {j for i in matched_pair_positions(tokens, classes) for j in (i, i + 1)}
    return [tok for j, tok in enumerate(tokens) if j not in paired]


def build_model(classes: list[str]):

    # one-hot: cada keyword é uma feature binária (presente/ausente), exceto as que foram concatenadas num par
    # + outro one-hot só para os pares de keywords que (quase) coincidem com o nome de uma classe
    singles = CountVectorizer(tokenizer=partial(unpaired_keywords, classes=classes), token_pattern=None, lowercase=False, binary=True)
    pairs   = CountVectorizer(tokenizer=partial(class_pairs, classes=classes), token_pattern=None, lowercase=False, binary=True)
    clf     = LogisticRegression(C=config.LR_C, class_weight='balanced', max_iter=3000)

    return make_pipeline(make_union(singles, pairs), clf)


def main():

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df  = pd.read_csv(config.TEST_PATH)

    X_train = train_df['keywords'].fillna('')
    X_test  = test_df['keywords'].fillna('')
    y       = train_df['medical_specialty']
    classes = sorted(y.unique())


    # ---- avaliação: cross-validation estratificada no train (o test não tem labels)
    cv   = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    pred = cross_val_predict(build_model(classes), X_train, y, cv=cv)

    print(f"Logistic regression, keywords + class-matching keyword pairs one-hot (C = {config.LR_C}, pair threshold > {config.MED_THRESHOLD}) | {config.CV_FOLDS}-fold CV\n")
    print(f"Accuracy:         {accuracy_score(y, pred):.4f}")
    print(f"Macro F1:         {f1_score(y, pred, average='macro'):.4f}")

    f1 = pd.Series(f1_score(y, pred, average=None, labels=classes), index=classes)
    print("\nPer-class F1:")
    print(f1.sort_values().round(4).to_string())

    below = f1[f1 < 0.25]
    print(f"\n{'FAIL' if len(below) else 'OK'}: {len(below)} class(es) with F1 below 25%{': ' + ', '.join(below.index) if len(below) else ''}")


    # ---- test -> results.txt (uma label por linha, mesma ordem que test_no_labels.csv, sem header)
    model = build_model(classes).fit(X_train, y)
    singles, pairs = (vec for _, vec in model[0].transformer_list)
    print(f"\nVocabulary: {len(singles.vocabulary_)} keywords + {len(pairs.vocabulary_)} pairs: {', '.join(sorted(pairs.vocabulary_))}")

    # linhas sem nenhuma feature ativa (ex: linha vazia ';;;' no test) -> só o intercept decidiria, e com
    # class_weight='balanced' o intercept não reflete a frequência das classes -> fallback para a classe mais comum
    most_common = y.mode()[0]
    no_features = model[0].transform(X_test).getnnz(axis=1) == 0

    test_pred = pd.Series(model.predict(X_test)).mask(no_features, most_common)
    test_pred.to_csv(config.RESULTS_PATH, index=False, header=False)

    print(f"Test: {len(test_pred)} predictions -> {config.RESULTS_PATH.name} ({no_features.sum()} without features -> '{most_common}')")



if __name__ == '__main__': main()
