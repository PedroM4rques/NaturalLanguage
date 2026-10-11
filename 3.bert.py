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


def build_clf():
    return LogisticRegression(C=config.BERT_LR_C, class_weight='balanced', max_iter=5000)


def main():

    train_df = pd.read_csv(config.TRAIN_PATH)
    y = train_df['medical_specialty']
    classes = sorted(y.unique())

    X_train = embed(build_text(train_df), 'train')
    print(f'Embeddings: {X_train.shape}')

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    pred = cross_val_predict(build_clf(), X_train, y, cv=cv)


    print(f"\nBERT embeddings ({config.BERT_MODEL}) + LR (C = {config.BERT_LR_C}) | "
          f"cols = {config.BERT_TEXT_COLS} | {config.CV_FOLDS}-fold CV\n")
    print(f"Accuracy:         {accuracy_score(y, pred):.4f}")
    print(f"Macro F1:         {f1_score(y, pred, average='macro'):.4f}")

    f1 = pd.Series(f1_score(y, pred, average=None, labels=classes), index=classes)
    print("\nPer-class F1:")
    print(f1.sort_values().round(4).to_string())

    below = f1[f1 < 0.25]
    print(f"\n{'FAIL' if len(below) else 'OK'}: {len(below)} class(es) with F1 below 25%"
          f"{': ' + ', '.join(below.index) if len(below) else ''}")

    out_dir = config.OUT_DIR / f'{make_tag()}_C{config.BERT_LR_C}'
    out_dir.mkdir(exist_ok=True)


    cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)
    cm.to_csv(out_dir / 'bert_confusion_matrix.csv')

    # ver onde errou
    errors = train_df.loc[pred != y, ['medical_specialty', 'description', 'sample_name']].copy()
    errors.insert(1, 'predicted', pred[pred != y])
    errors.to_csv(out_dir / 'bert_errors.csv', index=False)

    print(f"{len(errors)} erros -> outputs/bert_errors.csv")

    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average='macro')

    with open(out_dir / 'metrics.txt', 'w') as f:
        f.write(f"Accuracy:         {acc:.4f}\n")
        f.write(f"Macro F1:         {macro_f1:.4f}\n")

    if config.BERT_WRITE_RESULTS:
        test_df = pd.read_csv(config.TEST_PATH)
        X_test  = embed(build_text(test_df), 'test')
        clf     = build_clf().fit(X_train, y)
        out     = config.ROOT_DIR / 'results_bert.txt'
        config.restore_label_case(pd.Series(clf.predict(X_test))).to_csv(out, index=False, header=False)
        print(f"Test: {len(test_df)} predictions -> {out.name}")



if __name__ == '__main__': main()