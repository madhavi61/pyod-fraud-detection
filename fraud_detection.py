"""Train a PyOD autoencoder and evaluate fraud alerts on held-out transactions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 42
CONTAMINATION = 0.005
FEATURES = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]


def run_experiment(csv_path: Path) -> None:
    """Fit on unlabeled features and use Class only to evaluate the test set."""
    transactions = pd.read_csv(csv_path, usecols=[*FEATURES, "Class"])
    features = transactions[FEATURES].to_numpy(dtype=np.float64)
    labels = transactions["Class"].to_numpy()
    if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected finite features and binary Class labels.")

    # Split row indices without consulting labels. PyOD scales using training data.
    train_rows, test_rows = train_test_split(
        np.arange(len(transactions)), test_size=0.25, random_state=SEED
    )
    test_labels = labels[test_rows]
    if np.unique(test_labels).size != 2:
        raise ValueError("The test set must contain both transaction classes.")

    torch.set_num_threads(2)
    detector = AutoEncoder(
        contamination=CONTAMINATION,
        preprocessing=True,
        epoch_num=15,
        batch_size=1024,
        hidden_neuron_list=[32, 8],
        dropout_rate=0.0,
        device="cpu",
        random_state=SEED,
        verbose=0,
    )
    detector.fit(features[train_rows])
    scores = detector.decision_function(features[test_rows])

    # Apply the fitted training threshold to unseen reconstruction errors.
    alerts = (scores > detector.threshold_).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, alerts, labels=[0, 1]).ravel()

    print("PyOD AutoEncoder | Credit Card Fraud Detection")
    print(f"Total transactions: {len(transactions):,}  |  Features: {len(FEATURES)}")
    print(f"Training rows: {len(train_rows):,}  |  Test rows: {len(test_rows):,}")
    print(f"Actual fraud in test set: {test_labels.sum():,} ({test_labels.mean():.3%})")
    print(f"Training contamination: {CONTAMINATION:.2%}")
    print(f"Reconstruction error threshold: {detector.threshold_:.4f}")
    print()
    print(f"ROC AUC:           {roc_auc_score(test_labels, scores):.4f}")
    print(f"Average precision: {average_precision_score(test_labels, scores):.4f}")
    print(f"Alert precision:   {precision_score(test_labels, alerts, zero_division=0):.2%}")
    print(f"Fraud recall:      {recall_score(test_labels, alerts):.2%}")
    print()
    print(f"True negatives:  {tn:>6,}   False positives: {fp:>5,}")
    print(f"False negatives: {fn:>6,}   True positives:  {tp:>5,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to Kaggle's creditcard.csv")
    args = parser.parse_args()
    try:
        run_experiment(args.csv)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")


if __name__ == "__main__":
    main()
