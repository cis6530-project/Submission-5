#!/usr/bin/env python3
"""
Submission 5 — Deep Android Malware Detection CNN + Hyperparameter Search
"""
import os, re, glob, time
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

#config
DATA_DIR   = r"C:\Users\jdrob\OneDrive\Documents\Guelph\CIS6530_Intel\CIS_6530_Assignment_P4\opcodes"
EXPORT_DIR = r"C:\Users\jdrob\OneDrive\Documents\Guelph\CIS6530_Intel\CIS_6530_Assignment_P5\export_cnn_hyper"

os.makedirs(EXPORT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MIN_SAMPLES = 15
APT_NAME_RE = re.compile(r'^APT_([^_]+)_', re.IGNORECASE)

# ingestion from part 4
def parse_meta(fname):
    base = os.path.basename(fname)
    base_nosuf = base[:-7] if base.lower().endswith(".opcode") else base
    m = APT_NAME_RE.match(base_nosuf)
    return m.group(1) if m else "UNKNOWN_APT"


def read_opcodes(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip().lower() for ln in f if ln.strip()]

def ingest_opcodes(data_dir):
    print(f"[INGEST] Loading .opcode files…")
    paths = glob.glob(os.path.join(data_dir, "**", "*.opcode"), recursive=True)

    records, docs = [], []
    for fp in paths:
        ops = read_opcodes(fp)
        records.append({
            "path": fp,
            "filename": os.path.basename(fp),
            "apt": parse_meta(fp),
            "opcode_count": len(ops)
        })
        docs.append(" ".join(ops))

    df = pd.DataFrame(records)
    df["doc"] = docs

    df.to_csv(os.path.join(EXPORT_DIR, "file_summary.csv"), index=False)
    return df

# vocab and encoding 
def build_opcode_vocab(docs):
    counter = Counter()
    for d in docs:
        counter.update(d.split())

    opcode_to_id = {"<PAD>": 0, "<UNK>": 1}
    for idx, tok in enumerate(sorted(counter.keys(), key=lambda x: -counter[x]), start=2):
        opcode_to_id[tok] = idx

    print(f"[VOCAB] Size: {len(opcode_to_id)}")
    return opcode_to_id

def encode_doc(doc, vocab, max_len):
    ids = [vocab.get(tok, 1) for tok in doc.split()]
    if len(ids) > max_len:
        return ids[:max_len]
    else:
        return ids + [0] * (max_len - len(ids))

# dataset splitting
def build_cnn_dataset(df, max_len=2000):

    keep = df["apt"].value_counts()[lambda s: s >= MIN_SAMPLES].index
    df = df[df["apt"].isin(keep)].reset_index()

    docs = df["doc"].tolist()
    labels = df["apt"].tolist()

    vocab = build_opcode_vocab(docs)
    X = np.array([encode_doc(doc, vocab, max_len) for doc in docs])

    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    y = tf.keras.utils.to_categorical(y_int)

    X_trv, X_test, y_trv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y_int
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trv, y_trv, test_size=0.2, random_state=SEED,
        stratify=y_trv.argmax(axis=1)
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, vocab, le, len(le.classes_)


#og cnn from paper
def build_cnn_model(
    vocab_size, num_classes, max_len,
    emb_dim=8, 
    num_filters=64,  
    kernel_size=8, 
    dense_units=16,       
    dropout_rate=0.5,     
    l2_reg=1e-4   
):

    # EmbeddingLayer in Paper
    inputs = layers.Input(shape=(max_len,), dtype="int32")
    x = layers.Embedding(vocab_size, emb_dim)(inputs)

    # Convolution Module
    x = layers.Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)

    #  SpatialMaxPooling
    x = layers.GlobalMaxPooling1D()(x)

    # HIDDEN LAYER 
    x = layers.Dense(dense_units, activation="relu")(x)

    # DROPOUT
    x = layers.Dropout(dropout_rate)(x)

    # OUTPUT
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),  # ==== From Paper ====
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# one run
def run_cnn_single_combo(
    X_train, y_train, X_val, y_val, X_test, y_test,
    vocab_size, num_classes, max_len,
    emb_dim, num_filters, kernel_size,
    dense_units, dropout_rate,
    use_class_weight,
    epochs, batch_size
):
    # Class weighting
    class_weights = None
    if use_class_weight:
        y_int = np.argmax(y_train, axis=1)
        w = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_int),
            y=y_int
        )
        class_weights = {cls: weight for cls, weight in zip(np.unique(y_int), w)}

    model = build_cnn_model(
        vocab_size, num_classes, max_len,
        emb_dim=emb_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        class_weight=class_weights
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return acc, prec, rec, f1

# hyperparameter search
def hyperparameter_search(df):

    MAX_LEN = 2000
    X_train, y_train, X_val, y_val, X_test, y_test, vocab, le, num_classes = \
        build_cnn_dataset(df, max_len=MAX_LEN)

    vocab_size = len(vocab)

    # hyperparameter values    
    emb_dim_list       = [8, 32, 64]       # 8 is OG paper, class CNN uses deeper feature maps (no embedding)
    num_filters_list   = [32, 64, 128]  # class uses 16→32 filters; 64 matches paper
    kernel_list        = [3, 8]         # class CNN always used kernel 3, 8 paper, test bigger
    dense_list         = [16, 32]       # 16 = paper, 64 LSTM test, 32 to look at
    dropout_list       = [0.0, 0.5, 0.8]     # simple cnn had no dropout; paper 0.5, old code I checked 0.8
    class_weights_list = [True, False]  # was reccomended to balance to allow for generalization
    epochs_list        = [10, 20]           # Class example 3 not worth to test, paper 10, previous testing 20 
    batch_list         = [16]           # Stable google says not worth it to test for CNN


    results_csv = os.path.join(EXPORT_DIR, "cnn_hparam_results.csv")
    if not os.path.exists(results_csv):
        pd.DataFrame(columns=[
            "run_id",
            "emb_dim", "num_filters", "kernel_size",
            "class_weight", "dense_units", "dropout_rate",
            "epochs", "batch_size",
            "accuracy", "precision", "recall", "f1_score"
        ]).to_csv(results_csv, index=False)

    run_id = 0
    total_runs = (
        len(emb_dim_list) *
        len(num_filters_list) *
        len(kernel_list) *
        len(class_weights_list) *
        len(dense_list) *
        len(dropout_list) *
        len(epochs_list) *
        len(batch_list)
    )

    print(f"\n[INFO] Starting {total_runs} hyperparameter runs…")

    for emb_dim in emb_dim_list:
        for num_filters in num_filters_list:
            for kernel_size in kernel_list:
                for use_class_weight in class_weights_list:
                    for dense_units in dense_list:
                        for dropout_rate in dropout_list:
                            for epochs in epochs_list:
                                for batch_size in batch_list:

                                    run_id += 1
                                    print(f"\n==== RUN {run_id}/{total_runs} ====")
                                    print(
                                        f"emb_dim={emb_dim}, filters={num_filters}, kernel={kernel_size}, "
                                        f"class_weight={use_class_weight}, hidden={dense_units}, "
                                        f"dropout={dropout_rate}, epochs={epochs}, batch={batch_size}"
                                    )

                                    acc, prec, rec, f1 = run_cnn_single_combo(
                                        X_train, y_train,
                                        X_val, y_val,
                                        X_test, y_test,
                                        vocab_size, num_classes, MAX_LEN,
                                        emb_dim, num_filters, kernel_size,
                                        dense_units, dropout_rate,
                                        use_class_weight,
                                        epochs, batch_size
                                    )

                                    print("[RESULTS]")
                                    print(f"Accuracy : {acc:.4f}")
                                    print(f"Precision: {prec:.4f}")
                                    print(f"Recall   : {rec:.4f}")
                                    print(f"F1-score : {f1:.4f}")

                                    # Save row
                                    row = pd.DataFrame([{
                                        "run_id": run_id,
                                        "emb_dim": emb_dim,
                                        "num_filters": num_filters,
                                        "kernel_size": kernel_size,
                                        "class_weight": use_class_weight,
                                        "dense_units": dense_units,
                                        "dropout_rate": dropout_rate,
                                        "epochs": epochs,
                                        "batch_size": batch_size,
                                        "accuracy": acc,
                                        "precision": prec,
                                        "recall": rec,
                                        "f1_score": f1,
                                    }])

                                    row.to_csv(results_csv, mode="a", header=False, index=False)

    print(f"\n[COMPLETE] Results saved to: {results_csv}")

if __name__ == "__main__":
    df = ingest_opcodes(DATA_DIR)
    hyperparameter_search(df)
