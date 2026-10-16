from pathlib import Path

import pandas as pd


ROOT_DIR     = Path(__file__).resolve().parent
DATA_DIR     = ROOT_DIR / 'data'
RAW_DATA_DIR = ROOT_DIR / '_raw_data'

RAW_TRAIN_PATH = RAW_DATA_DIR / 'train.csv'
RAW_TEST_PATH  = RAW_DATA_DIR / 'test_no_labels.csv'

TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH  = DATA_DIR / 'test.csv'

RESULTS_PATH = ROOT_DIR / 'results.txt'

# 1.rules_based

MED_THRESHOLD   = 0.90   # score (0-1) que uma keyword (ou par de keywords) tem de ultrapassar (>) para o matcher (min edit distance) aceitar a classe
DROP_FRAC       = 0.05    # fração de keywords menos informativas a remover no data augmentation

# 2.log-regression ############

_MED_THRESHOLD  = MED_THRESHOLD
LR_C            = 0.015 # 1/L2_regularization
CV_FOLDS        = 5
SEED            = 0

# 3.BERT

BERT_MODELS = {
    'clinicalbert': dict(name='emilyalsentzer/Bio_ClinicalBERT', max_len=512, chunk_words=250, lr_c=50),   # https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT, cased
    'spubmedbert': dict(name='pritamdeka/S-PubMedBert-MS-MARCO', max_len=512, chunk_words=250, lr_c=50),   # https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO
    'biomedbert': dict(name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', max_len=512, chunk_words=250, lr_c=50),   # https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltex, uncased
    'biobert': dict(name='dmis-lab/biobert-base-cased-v1.2', max_len=512, chunk_words=250, lr_c=50),   # https://huggingface.co/dmis-lab/biobert-base-cased-v1.2, cased
    'mpnet': dict(name='sentence-transformers/all-mpnet-base-v2', max_len=384, chunk_words=200, lr_c=50),   # genérico, https://huggingface.co/sentence-transformers/all-mpnet-base-v2
}

BERT_CHOICE = 'biomedbert' # mudar o nome para o embedding que se quer

CACHE_DIR = DATA_DIR / '.cache'   # cache do embeddings
OUT_DIR   = ROOT_DIR / 'outputs'  # matriz de confusao e erros
_bert            = BERT_MODELS[BERT_CHOICE]
BERT_MODEL       = _bert['name']
BERT_MAX_LEN     = _bert['max_len']        # max de tokens
BERT_CHUNK_WORDS = _bert['chunk_words']    # palavras por chunk
BERT_LR_C        = _bert['lr_c']           # 1/L2

BERT_TEXT_COLS     = ['description', 'sample_name', 'transcription']
# BERT_TEXT_COLS     = ['description', 'sample_name', 'transcription', 'keywords']
# BERT_TEXT_COLS     = ['description', 'transcription']
# BERT_TEXT_COLS     = ['transcription']
# BERT_TEXT_COLS     = ['keywords']
# BERT_TEXT_COLS     = ['description']
# BERT_TEXT_COLS     = ['sample_name']

BERT_BATCH         = 16
BERT_WRITE_RESULTS = False  # True -> escreve results_bert.txt

BERT_ENSEMBLE        = ['clinicalbert', 'biomedbert', 'spubmedbert', 'mpnet', 'biobert']
BERT_ENSEMBLE_METHOD = 'max'    # 'mean' (media das probabilidades) ou 'max' (o modelo mais confiante)
...


# aux functions

def restore_label_case(pred: pd.Series) -> pd.Series:

    raw_labels = pd.read_csv(RAW_TRAIN_PATH, sep=';')['medical_specialty'].dropna().str.strip()
    mapping    = {label.lower(): label for label in raw_labels}

    return pred.map(mapping)


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, classes: list[str], title: str, filename: str) -> None:

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm   = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes)
    norm = cm.div(cm.sum(axis=1), axis=0)   # cor = fracao da linha (recall), senao a surgery apaga as outras classes

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm.values, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fraction of true class')

    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm.iat[i, j]
            if v: ax.text(j, i, v, ha='center', va='center', fontsize=8, color='white' if norm.iat[i, j] > 0.5 else '#333')

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Confusion matrix -> outputs/{filename}")
