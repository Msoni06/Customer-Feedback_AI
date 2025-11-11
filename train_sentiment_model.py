import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

print("Loading cleaned data...")
try:
    df = pd.read_csv("cleaned_feedback.csv")
except FileNotFoundError:
    print("Error: cleaned_feedback.csv not found.")
    print("Please run data_preprocessing.py first.")
    exit()

labels_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df = df.dropna(subset=['feedback'])
df["label"] = df["sentiment"].map(labels_map)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["feedback"], padding=True, truncation=True, max_length=128)

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
    label2id=labels_map
).to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="./sentiment_model_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("\n--- Evaluation Results ---")
eval_results = trainer.evaluate()
print(eval_results)

final_model_path = "./sentiment_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"Model saved to {final_model_path}")