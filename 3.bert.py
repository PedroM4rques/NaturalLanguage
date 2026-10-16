import numpy as np
import pandas as pd
import torch

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import config


def build_text(df: pd.DataFrame) -> pd.Series:
    return df[config.BERT_TEXT_COLS].fillna('').astype(str).agg(' . '.join, axis=1)


def chunk(text: str, size: int) -> list[str]:
    words = text.split()
    if not words:
        return ['']
    return [' '.join(words[i:i + size]) for i in range(0, len(words), size)]


def make_tag() -> str:
    return f"{config.BERT_CHOICE.replace('/', '_')}_{'-'.join(config.BERT_TEXT_COLS)}_{config.BERT_CHUNK_WORDS}"


def use_model(key: str):
    m = config.BERT_MODELS[key]
    config.BERT_CHOICE      = key
    config.BERT_MODEL       = m['name']
    config.BERT_MAX_LEN     = m['max_len']
    config.BERT_CHUNK_WORDS = m['chunk_words']
    config.BERT_LR_C        = m['lr_c']


def embed(texts: pd.Series, name: str) -> np.ndarray:

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # usar GPU ou CPU dependendo da maquina do user
    
    tag = make_tag()
    cache = config.CACHE_DIR / f'{name}_{tag}.npy'

    if cache.exists():
        print(f'[cache] {cache.name}')
        return np.load(cache)

    print(f'A calcular embeddings de {len(texts)} textos ({name}) usando {device}...')

    model = SentenceTransformer(config.BERT_MODEL, device=device)
    model.max_seq_length = config.BERT_MAX_LEN

    chunks, doc_ids = [], []
    for i, t in enumerate(texts):
        parts = chunk(t, config.BERT_CHUNK_WORDS)
        chunks.extend(parts)
        doc_ids.extend([i] * len(parts))

    vecs = model.encode(
        chunks, batch_size=config.BERT_BATCH, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
        )

    doc_ids = np.array(doc_ids)
    X = np.zeros((len(texts), vecs.shape[1]))
    np.add.at(X, doc_ids, vecs)
    X /= np.bincount(doc_ids, minlength=len(texts))[:, None]

    config.CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache, X)
    return X


def ensemble_proba(train_df, y, cv, test_df=None):
    P_cv, P_test = [], []

    for key in config.BERT_ENSEMBLE:
        use_model(key)
        X = embed(build_text(train_df), 'train')

        P = cross_val_predict(build_clf(), X, y, cv=cv, method='predict_proba')
        P_cv.append(P)
        print(f"  {key:14s} acc {accuracy_score(y, np.sort(y.unique())[P.argmax(axis=1)]):.4f}")

        if test_df is not None:
            X_test = embed(build_text(test_df), 'test')
            P_test.append(build_clf().fit(X, y).predict_proba(X_test))

    combine = np.max if config.BERT_ENSEMBLE_METHOD == 'max' else np.mean
    P_cv    = combine(np.stack(P_cv), axis=0)
    P_test  = combine(np.stack(P_test), axis=0) if P_test else None

    return P_cv, P_test



def build_clf():
    return LogisticRegression(C=config.BERT_LR_C, class_weight='balanced', max_iter=5000)


def main():

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df  = pd.read_csv(config.TEST_PATH) if config.BERT_WRITE_RESULTS else None
    y        = train_df['medical_specialty']
    classes  = np.array(sorted(y.unique()))      # mesma ordem das colunas do predict_proba

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


    if config.BERT_ENSEMBLE:
        print(f"Ensemble ({config.BERT_ENSEMBLE_METHOD}) de {config.BERT_ENSEMBLE}:")
        P, P_test = ensemble_proba(train_df, y, cv, test_df)
        pred      = classes[P.argmax(axis=1)]
        test_pred = classes[P_test.argmax(axis=1)] if P_test is not None else None
        run_tag   = f"ensemble-{config.BERT_ENSEMBLE_METHOD}_{'+'.join(config.BERT_ENSEMBLE)}_{'-'.join(config.BERT_TEXT_COLS)}"

    else:
        use_model(config.BERT_CHOICE)
        X_train = embed(build_text(train_df), 'train')
        pred    = cross_val_predict(build_clf(), X_train, y, cv=cv)
        test_pred = build_clf().fit(X_train, y).predict(embed(build_text(test_df), 'test')) if test_df is not None else None
        run_tag = f'{make_tag()}_C{config.BERT_LR_C}'


    acc      = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average='macro')

    print(f"\n{run_tag} | {config.CV_FOLDS}-fold CV\n")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")

    f1 = pd.Series(f1_score(y, pred, average=None, labels=classes), index=classes)
    print("\nPer-class F1:")
    print(f1.sort_values().round(4).to_string())

    below = f1[f1 < 0.25]
    print(f"\n{'FAIL' if len(below) else 'OK'}: {len(below)} class(es) with F1 below 25%"
          f"{': ' + ', '.join(below.index) if len(below) else ''}")


    out_dir = config.OUT_DIR / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)
    cm.to_csv(out_dir / 'bert_confusion_matrix.csv')

    # ver onde errou
    errors = train_df.loc[pred != y, ['medical_specialty', 'description', 'sample_name']].copy()
    errors.insert(1, 'predicted', pred[pred != y])
    errors.to_csv(out_dir / 'bert_errors.csv', index=False)

    with open(out_dir / 'metrics.txt', 'w') as f:
        f.write(f"Accuracy:         {acc:.4f}\n")
        f.write(f"Macro F1:         {macro_f1:.4f}\n")

    print(f"{len(errors)} erros -> {out_dir.name}/")


    if test_pred is not None:
        out = out_dir / 'results_bert.txt'
        config.restore_label_case(pd.Series(test_pred)).to_csv(out, index=False, header=False)
        print(f"Test: {len(test_pred)} predictions -> {out_dir.name}/{out.name}")


if __name__ == '__main__': main()