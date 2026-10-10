## Requirements

- Python 3.14
- pandas 3.0.6
- numpy 2.5.3
- scikit-learn 1.9.1
- rapidfuzz 3.14.6
- jupyter (to run the cleaning notebook) and matplotlib (only for the exploration notebook)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas==3.0.6 numpy==2.5.3 scikit-learn==1.9.1 rapidfuzz==3.14.6 jupyter matplotlib
```


## How to run

Run every command from the project root, except step 1.

**1. Clean the raw data.** 

The notebook uses relative paths, so run it from inside `_raw_data/`.

```bash
cd _raw_data
jupyter nbconvert --to notebook --execute clean_data.ipynb --output clean_data.ipynb
cd ..
```

**2. Augment the keywords.**

```bash
python data/data_augmentation.py
```

This step rewrites `data/train.csv` and `data/test.csv` in place. Always run step 1 first, so it starts from freshly cleaned data.

**3. Run the models.**

```bash
python 1.rules_based+fallback.py
python 2.log_regression.py
python 3.*.py
```

> **The best model is `2.log_regression.py`.** 

> **All scripts write `results.txt`.** 

## Hyperparameters

See `config.py`.

