# Credit Card Fraud Detection with PyOD

This project uses PyOD's AutoEncoder to flag unusual credit card transactions. The model trains on transaction features without fraud labels. Transactions with reconstruction errors above a threshold learned from the training data are flagged for review. Fraud labels from the held-out test set are used to evaluate the results.

## Run on macOS

1. Install Python 3.12 and open a terminal in the project folder.
2. Download the [dataset ZIP](https://www.kaggle.com/api/v1/datasets/download/whenamancodes/fraud-detection) from the [Kaggle Fraud Detection dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection). Unzip it and place `creditcard.csv` at `data/creditcard.csv`. Create the `data` folder if needed.
3. Create a virtual environment, install the dependencies, and run the script.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fraud_detection.py data/creditcard.csv
```

The dataset and local virtual environment are excluded from version control.

## Experiment Results

The macOS run evaluated 71,202 held-out transactions, including 113 confirmed fraud cases.

| Metric | Result |
| --- | ---: |
| Reconstruction-error threshold | 8.9113 |
| ROC AUC | 0.9342 |
| Average precision | 0.0832 |
| Alert precision | 10.56% |
| Fraud recall | 33.63% |

The model flagged 38 fraud cases, missed 75, and flagged 322 legitimate transactions. The script runs on CPU with a fixed random seed. Results can vary slightly across platforms.
