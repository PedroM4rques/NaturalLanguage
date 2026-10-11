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

BERT_CHOICE = 'spubmedbert' # mudar o nome para o embedding que se quer

CACHE_DIR = DATA_DIR / '.cache'   # cache do embeddings
OUT_DIR   = ROOT_DIR / 'outputs'  # matriz de confusao e erros
_bert            = BERT_MODELS[BERT_CHOICE]
BERT_MODEL       = _bert['name']
BERT_MAX_LEN     = _bert['max_len']        # max de tokens
BERT_CHUNK_WORDS = _bert['chunk_words']    # palavras por chunk
BERT_LR_C        = _bert['lr_c']           # 1/L2

# BERT_TEXT_COLS     = ['description', 'sample_name', 'transcription']
# BERT_TEXT_COLS     = ['description', 'transcription']
BERT_TEXT_COLS     = ['transcription']
# BERT_TEXT_COLS     = ['description']
# BERT_TEXT_COLS     = ['sample_name']

BERT_BATCH         = 16
BERT_WRITE_RESULTS = False  # True -> escreve results_bert.txt
...


# aux functions

def restore_label_case(pred: pd.Series) -> pd.Series:
    """Mapeia as labels lowercased (clean_data) de volta para a capitalização original do raw train (ex: 'general medicine' -> 'General Medicine')."""

    raw_labels = pd.read_csv(RAW_TRAIN_PATH, sep=';')['medical_specialty'].dropna().str.strip()
    mapping    = {label.lower(): label for label in raw_labels}

    return pred.map(mapping)
