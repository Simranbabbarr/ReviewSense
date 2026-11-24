import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

DATA_FILE = "feedback student.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "reviewsense_model"
MAX_LENGTH = 192
BATCH_SIZE = 8
EPOCHS = 6
LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.03
PATIENCE = 2
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Keep '{DATA_FILE}' in the same folder as train.py.")

    try:
        df = pd.read_csv(DATA_FILE, encoding="cp1252")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_FILE, encoding="utf-8")

    if not {"comment", "quality"}.issubset(df.columns):
        raise ValueError("The CSV must contain 'comment' and 'quality' columns.")

    df = df[["comment", "quality"]].dropna().copy()
    df["comment"] = df["comment"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["quality"] = df["quality"].astype(str).str.strip().str.lower()
    df["normalized_comment"] = df["comment"].str.lower()

    placeholders = {"", "no comments", "none", "n/a", "na", "null", "nan"}
    df = df[~df["normalized_comment"].isin(placeholders)].copy()

    label_counts_per_comment = df.groupby("normalized_comment")["quality"].nunique()
    conflicting_comments = label_counts_per_comment[label_counts_per_comment > 1].index
    print(f"Conflicting review texts removed: {len(conflicting_comments)}")
    df = df[~df["normalized_comment"].isin(conflicting_comments)].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["normalized_comment"], keep="first")
    print(f"Duplicate review rows removed: {before - len(df)}")
    df = df.reset_index(drop=True)

    print(f"Usable unique reviews: {len(df)}")
    print("\nClass distribution:")
    print(df["quality"].value_counts().to_string())

    return df[["comment", "quality"]]

def evaluate(model, loader, device, criterion):
    model.eval()
    actual = []
    predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
            actual.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(actual, predictions),
        "macro_f1": f1_score(actual, predictions, average="macro"),
        "actual": actual,
        "predictions": predictions
    }

def main():
    set_seed(SEED)
    df = load_data()

    labels = sorted(df["quality"].unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    texts = df["comment"].tolist()
    y = df["quality"].map(label_to_id).tolist()

    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, y, test_size=0.20, random_state=SEED, stratify=y
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.1875,
        random_state=SEED, stratify=train_val_labels
    )

    print(f"\nTraining reviews: {len(train_texts)}")
    print(f"Validation reviews: {len(val_texts)}")
    print(f"Test reviews: {len(test_texts)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data = ReviewDataset(train_texts, train_labels, tokenizer)
    val_data = ReviewDataset(val_texts, val_labels, tokenizer)
    test_data = ReviewDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    counts = np.bincount(train_labels, minlength=len(labels))
    class_weights = (len(train_labels) / (len(labels) * counts)) ** 0.5
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    print("\nTraining class weights:")
    for label, weight in zip(labels, class_weights):
        print(f"{label}: {weight:.3f}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.10),
        num_training_steps=total_steps
    )

    best_val_f1 = -1.0
    best_epoch = 0
    wait = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = criterion(outputs.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)} | Loss {loss.item():.4f}"
                )

        val_metrics = evaluate(model, val_loader, device, criterion)
        train_loss = total_loss / max(len(train_loader), 1)

        print(
            f"\nEpoch {epoch + 1}: train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, criterion)

    print(f"\nBest checkpoint: epoch {best_epoch}")
    print("\nFINAL HELD-OUT TEST RESULTS")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {test_metrics['macro_f1']:.4f}")
    print("\nClassification Report")
    print(classification_report(
        test_metrics["actual"], test_metrics["predictions"],
        labels=list(range(len(labels))), target_names=labels, zero_division=0
    ))
    print("Confusion Matrix")
    print(confusion_matrix(test_metrics["actual"], test_metrics["predictions"]))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = {
        "accuracy": float(test_metrics["accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "best_epoch": int(best_epoch),
        "labels": labels,
        "class_weights": {label: float(weight) for label, weight in zip(labels, class_weights)},
        "test_reviews": len(test_texts),
        "seed": SEED
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nModel and metrics saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
