import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = "cleaned_data"
SPLITS = ["train", "test"] 
MODES = [
    "birth",
    "extrovert",
    "feeling",
    "gender",
    "judging",
    "nationality",
    "political",
    "sensing"
]

TEXT_COLUMNS = ["post", "post_masked"]
RANDOM_STATE = 1234

def load_split(mode, split):
    path = os.path.join(BASE_DIR, split, f"{mode}_{split}.csv")
    return pd.read_csv(path)

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    return acc, f1, prec, rec

for mode in MODES:
    print("\n" + "=" * 50)
    print(f"Evaluating TF-IDF for {mode.upper()}")
    print("=" * 50)

    train_df = load_split(mode, "train")
    test_df = load_split(mode, "test")

    label_col = [c for c in train_df.columns if c not in TEXT_COLUMNS][0]


    y_train = train_df[label_col]
    y_test = test_df[label_col]


    for text_col in TEXT_COLUMNS:
        X_train = train_df[text_col].fillna("").astype(str)
        X_test = test_df[text_col].fillna("").astype(str)

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=50000
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc, f1, prec, rec = evaluate(y_test, y_pred)

        print(
            f"Result for {mode} ({text_col}): "
            f"Accuracy={acc:.4f}, "
            f"Precision={prec:.4f}, "
            f"Recall={rec:.4f}"
            f"F1={f1:.4f}, "
        )
