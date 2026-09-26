import pandas as pd
from rapidfuzz import fuzz, process

from sklearn.metrics import accuracy_score, f1_score

import config


def fuzzy_match(keywords: str, classes: list[str]) -> tuple[str | None, float]:

    if pd.isna(keywords):
        return None, 0.0

    tokens = [kw for kw in keywords.split(',') if kw]

    # cada keyword + cada par de keywords seguidas ('cardiovascular,pulmonary' -> 'cardiovascularpulmonary')
    candidates = []
    for i, tok in enumerate(tokens):
        candidates.append(tok)
        if i + 1 < len(tokens):
            candidates.append(tok + tokens[i + 1])

    # melhor score; em empate, a keyword que aparece primeiro
    best_class, best_score = None, 0.0
    for cand in candidates:
        cls, score, _ = process.extractOne(cand, classes, scorer=fuzz.ratio)
        if score / 100 > best_score:
            best_class, best_score = cls, score / 100

    return best_class, best_score


def predict(df: pd.DataFrame, classes: list[str], fallback: str) -> pd.DataFrame:

    matches = df['keywords'].apply(lambda kw: fuzzy_match(kw, classes))

    out = pd.DataFrame({'match': matches.str[0], 'score': matches.str[1]}, index=df.index)
    out['matched'] = out['score'] >= config.FUZZY_THRESHOLD
    out['pred']    = out['match'].where(out['matched'], fallback)

    return out


def main():

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df  = pd.read_csv(config.TEST_PATH)

    classes      = sorted(train_df['medical_specialty'].unique())
    most_common  = train_df['medical_specialty'].mode()[0]


    y   = train_df['medical_specialty']
    out = predict(train_df, classes, most_common)

    print(f"Fuzzy matcher, every keyword (threshold = {config.FUZZY_THRESHOLD}) | fallback = '{most_common}'\n")
    print(f"Accuracy:         {accuracy_score(y, out['pred']):.4f}")
    print(f"Macro F1:         {f1_score(y, out['pred'], average='macro'):.4f}")
    print(f"Coverage (match): {out['matched'].mean():.1%}  ({out['matched'].sum()}/{len(out)})")
    print(f"Match precision:  {(out.loc[out['matched'], 'pred'] == y[out['matched']]).mean():.1%}")

    f1 = pd.Series(f1_score(y, out['pred'], average=None, labels=classes), index=classes)
    print("\nPer-class F1:")
    print(f1.sort_values().round(4).to_string())

    below = f1[f1 < 0.25]
    print(f"\n{'FAIL' if len(below) else 'OK'}: {len(below)} class(es) with F1 below 25%{': ' + ', '.join(below.index) if len(below) else ''}")


    # ---- test -> results.txt (uma label por linha, mesma ordem que test_no_labels.csv, sem header)
    test_out = predict(test_df, classes, most_common)
    test_out['pred'].to_csv(config.RESULTS_PATH, index=False, header=False)

    print(f"\nTest: {test_out['matched'].sum()}/{len(test_out)} matched -> {config.RESULTS_PATH.name} ({len(test_out)} lines)")



if __name__ == '__main__': main()
