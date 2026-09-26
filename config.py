from pathlib import Path


ROOT_DIR     = Path(__file__).resolve().parent
DATA_DIR     = ROOT_DIR / 'data'
RAW_DATA_DIR = ROOT_DIR / '_raw_data'

RAW_TRAIN_PATH = RAW_DATA_DIR / 'train.csv'
RAW_TEST_PATH  = RAW_DATA_DIR / 'test_no_labels.csv'

TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH  = DATA_DIR / 'test.csv'

RESULTS_PATH = ROOT_DIR / 'results.txt'

FUZZY_THRESHOLD = 0.7    # score mínimo (0-1) para o fuzzy matcher atribuir logo a classe (afinado em 0.40-1.00, ver testes)

DROP_FRAC = 0.05    # fração de keywords menos informativas a remover no data augmentation
