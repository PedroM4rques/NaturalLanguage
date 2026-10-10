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
LR_C            = 0.015 # inverso da força de regularização (L2) da logistic regression (grid search CV macro F1; plateau entre 0.01 e 0.07)
CV_FOLDS        = 5
SEED            = 0

# 3.TODO.py

...


# aux functions

def restore_label_case(pred: pd.Series) -> pd.Series:
    """Mapeia as labels lowercased (clean_data) de volta para a capitalização original do raw train (ex: 'general medicine' -> 'General Medicine')."""

    raw_labels = pd.read_csv(RAW_TRAIN_PATH, sep=';')['medical_specialty'].dropna().str.strip()
    mapping    = {label.lower(): label for label in raw_labels}

    return pred.map(mapping)
