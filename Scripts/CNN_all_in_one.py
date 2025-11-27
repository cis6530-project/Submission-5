#!/usr/bin/env python3
"""
Submission 5 - Deep Android Malware Detection-style CNN
"""
import os, re, glob, time
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

import matplotlib.pyplot as plt
import seaborn as sns


#config
DATA_DIR   = r"C:\Users\jdrob\OneDrive\Documents\Guelph\CIS6530_Intel\CIS_6530_Assignment_P4\opcodes"
EXPORT_DIR = r"C:\Users\jdrob\OneDrive\Documents\Guelph\CIS6530_Intel\CIS_6530_Assignment_P5\export_cnn"

os.makedirs(EXPORT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MIN_SAMPLES = 15    # same as S4 but can try if enough time to test with less represented samples
TOKEN_PATTERN = r"\S+"
APT_NAME_RE = re.compile(r'^APT_([^_]+)_', re.IGNORECASE)

USE_CLASS_WEIGHTS = True   # toggle ON/OFF for class weights

# A 4 ingestion
def parse_meta(fname: str) -> str:
    base = os.path.basename(fname)
    base_nosuf = base[:-7] if base.lower().endswith(".opcode") else base
    m = APT_NAME_RE.match(base_nosuf)
    if m:
        return m.group(1).strip()
    return "UNKNOWN_APT"


def get_filetype(fname: str) -> str:
    base = os.path.basename(fname)
    if base.lower().endswith(".opcode"):
        base = base[:-7]
    if "." in base:
        return base.rsplit(".", 1)[-1].lower()
    return "unknown"


def read_opcodes(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        ops = [ln.strip().lower() for ln in f if ln.strip()]
    return ops


def ingest_opcodes(data_dir: str) -> pd.DataFrame:
    print(f"\n[INGEST] Scanning for .opcode files under: {data_dir}")
    records, docs = [], []

    all_paths = glob.glob(os.path.join(data_dir, "**", "*.opcode"), recursive=True)
    print(f"[INGEST] Found {len(all_paths)} .opcode files.")

    for i, fp in enumerate(all_paths, 1):
        try:
            ops = read_opcodes(fp)
        except Exception as e:
            print(f"[WARN] Failed reading {fp}: {e}")
            continue

        rec = {
            "path": fp,
            "filename": os.path.basename(fp),
            "apt": parse_meta(fp),
            "filetype": get_filetype(fp),
            "size_bytes": os.path.getsize(fp),
            "opcode_count": len(ops),
            "unique_opcodes": len(set(ops))
        }
        records.append(rec)
        docs.append(" ".join(ops))

        if i % 50 == 0:
            print(f"[INGEST] Processed {i}/{len(all_paths)} files...")

    df = pd.DataFrame(records)
    df["doc"] = docs

    raw_summary = os.path.join(EXPORT_DIR, "file_summary_submission5.csv")
    df.to_csv(raw_summary, index=False)
    print(f"[SAVE] file_summary_submission5.csv → {raw_summary}")

    print("\n[INGEST] Summary head:")
    print(df.head(3))

    return df

#dataset creation
def build_opcode_vocab(docs, min_freq: int = 1):
    counter = Counter()
    for doc in docs:
        counter.update(doc.split())

    vocab = [tok for tok, c in counter.items() if c >= min_freq]
    vocab.sort(key=lambda t: (-counter[t], t))

    opcode_to_id = {"<PAD>": 0, "<UNK>": 1}
    for i, tok in enumerate(vocab, start=2):
        opcode_to_id[tok] = i

    print(f"[VOCAB] Size={len(opcode_to_id)} (PAD+UNK included)")
    return opcode_to_id

def encode_doc(doc: str, opcode_to_id, max_len: int):
    tokens = doc.split()
    ids = [opcode_to_id.get(t, 1) for t in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))
    return ids

##########################################work on these fixes later
def build_cnn_dataset(df: pd.DataFrame,
                      max_len: int = 2000, 
                      min_class_samples: int = 15,
                      test_size: float = 0.2,
                      val_size: float = 0.2):

    print(f"\n[CLASS FILTER] Removing classes with < {min_class_samples} samples...")
    counts = df["apt"].value_counts()
    keep = counts[counts >= min_class_samples].index
    df = df[df["apt"].isin(keep)].reset_index(drop=True)
    print(df["apt"].value_counts())

    docs = df["doc"].tolist()
    labels = df["apt"].tolist()

    opcode_to_id = build_opcode_vocab(docs)
    X_all = np.array([encode_doc(d, opcode_to_id, max_len) for d in docs], dtype="int32")

    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    y_all = tf.keras.utils.to_categorical(y_int)

    # FIRST SPLIT (train+val vs test)
    X_train_val, X_test, y_train_val, y_test, y_int_train_val, y_int_test = train_test_split(
        X_all, y_all, y_int, test_size=test_size, random_state=SEED, stratify=y_int
    )

    # SECOND SPLIT (train vs val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=SEED,
        stratify=y_train_val.argmax(axis=1)
    )

    print(f"[SPLIT] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        opcode_to_id,
        le,
        len(le.classes_)
    )

# CNN MODEL
def build_cnn_model(vocab_size: int,
                    num_classes: int,
                    max_len: int,
                    emb_dim: int = 64,  # change imbedding from 8 (64% accuracy) to 32 (76% accuracy) or 64 (81% accuracy) 2000 max len 
                    conv_filters: int = 128,
                    conv_kernel: int = 8,
                    hidden_units: int = 32,
                    l2_reg: float = 1e-4,
                    dropout_rate: float = 0):

    inputs = layers.Input(shape=(max_len,), dtype="int32")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        input_length=max_len,
        name="opcode_embedding"
    )(inputs)

    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv1"
    )(x)

    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(
        hidden_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)

    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n[MODEL] Summary:")
    model.summary()

    return model

#cnn training
def train_and_evaluate_cnn():
    df = ingest_opcodes(DATA_DIR)
    MAX_LEN = 2000
    (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        opcode_to_id,
        label_encoder,
        num_classes
    ) = build_cnn_dataset(df, max_len=MAX_LEN)

    # class weight tests
    
    print("\n[DEBUG] Computing class weights...")
    if USE_CLASS_WEIGHTS:
        y_train_int = np.argmax(y_train, axis=1)
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_int),
            y=y_train_int
        )

        class_weights = {i: w for i, w in enumerate(class_weights_array)}
        print("[DEBUG] Class weights:", class_weights)
    else:
        class_weights = None
        print("\n[DEBUG] Class weights disabled.")    

    #-----------

    vocab_size = len(opcode_to_id)
    model = build_cnn_model(vocab_size, num_classes, MAX_LEN)

    print("\n[DEBUG] Exporting vocabulary, sequence lengths, class distribution...")

    # --- CLASS DISTRIBUTION CSV ---
    class_dist_path = os.path.join(EXPORT_DIR, "cnn_class_distribution.csv")
    pd.DataFrame(df["apt"].value_counts()).to_csv(class_dist_path)
    print(f"[DEBUG] Saved → {class_dist_path}")

    # --- VOCAB FREQUENCY ---
    vocab_counts = Counter(" ".join(df["doc"]).split())
    vocab_df = pd.DataFrame({"token": list(vocab_counts.keys()),
                             "count": list(vocab_counts.values())})
    vocab_df.sort_values("count", ascending=False).to_csv(
        os.path.join(EXPORT_DIR, "cnn_vocab_frequency.csv"), index=False)
    print("[DEBUG] Saved vocab freq → cnn_vocab_frequency.csv")

    vocab_df.head(50).to_csv(os.path.join(EXPORT_DIR, "cnn_vocab_top50.csv"), index=False)
    vocab_df[vocab_df["count"] <= 2].to_csv(os.path.join(EXPORT_DIR, "cnn_vocab_low_frequency.csv"), index=False)

    # --- SEQUENCE LENGTH DISTRIBUTION ---
    seq_lengths = [len(doc.split()) for doc in df["doc"]]
    seq_df = pd.DataFrame({"seq_len": seq_lengths})
    seq_stats_path = os.path.join(EXPORT_DIR, "cnn_sequence_length_stats.csv")
    seq_df.to_csv(seq_stats_path, index=False)
    print(f"[DEBUG] Saved seq length stats → {seq_stats_path}")
    print(seq_df.describe())

    print(f"[DEBUG] Sequences truncated (> {MAX_LEN}): {sum(l > MAX_LEN for l in seq_lengths)}")

    # --- SAVE MODEL SUMMARY ---
    summary_file = os.path.join(EXPORT_DIR, "cnn_model_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"[DEBUG] Saved model summary → {summary_file}")

    # Early stopping & LR schedule
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=1, min_lr=1e-5, verbose=1
        ),
    ]

    BATCH = 16
    EPOCHS = 20

    print("\n[TRAIN] Starting CNN training...")
    t0 = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=2,
        class_weight=class_weights if USE_CLASS_WEIGHTS else None # class weights toggle test
    )

    print(f"[TRAIN] Completed in {time.time()-t0:.2f} sec")
    hist_path = os.path.join(EXPORT_DIR, "cnn_training_history.csv")
    pd.DataFrame(history.history).to_csv(hist_path, index=False)
    print(f"[DEBUG] Saved training history → {hist_path}")

    print("\n[TEST] Evaluating...")
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    print("\n[RESULTS] CNN Metrics")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # RAW PREDICTIONS CSV
    pred_df = pd.DataFrame({
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_label": [label_encoder.classes_[i] for i in y_true],
        "pred_label": [label_encoder.classes_[i] for i in y_pred]
    })
    pred_csv = os.path.join(EXPORT_DIR, "cnn_raw_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"[DEBUG] Saved raw predictions → {pred_csv}")

    # CONFUSION MATRIX CSV & PNG
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_df.to_csv(os.path.join(EXPORT_DIR, "cnn_confusion_matrix.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, "cnn_confusion_matrix.png"), dpi=150)
    plt.close()

    # Classification report CSV
    cr_df = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=label_encoder.classes_,
        zero_division=0, output_dict=True)).transpose()
    cr_df.to_csv(os.path.join(EXPORT_DIR, "cnn_classification_report.csv"))

    # Filter activation diagnostics
    conv_layer = model.get_layer("conv1")
    conv_probe = tf.keras.Model(model.input, conv_layer.output)
    conv_out = conv_probe.predict(X_test[:32])
    filter_means = conv_out.mean(axis=(0, 1))

    pd.DataFrame({
        "filter_idx": range(len(filter_means)),
        "mean_activation": filter_means
    }).to_csv(os.path.join(EXPORT_DIR, "cnn_filter_activation_stats.csv"), index=False)

    print("[DEBUG] Saved conv filter activation stats.")

    # Save model + vocab + mapping
    model.save(os.path.join(EXPORT_DIR, "cnn_apt_model.h5"))

    with open(os.path.join(EXPORT_DIR, "cnn_opcode_vocab.txt"), "w") as f:
        for tok, idx in sorted(opcode_to_id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{tok}\n")

    with open(os.path.join(EXPORT_DIR, "cnn_label_mapping.txt"), "w") as f:
        for idx, label in enumerate(label_encoder.classes_):
            f.write(f"{idx}\t{label}\n")

    print("\n[SAVE] CNN model + vocab + labels saved.")

    return acc, precision, recall, f1

if __name__ == "__main__":
    train_and_evaluate_cnn()
