import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load CSV
df = pd.read_csv("data.csv")
df = df.dropna(subset=['text', 'label'])

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

X_train, y_train = train_df['text'], train_df['label']
X_val, y_val = val_df['text'], val_df['label']
X_test, y_test = test_df['text'], test_df['label']

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=3000,
        solver='lbfgs',
        random_state=42,
        verbose=1
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
def evaluate(model, X, y, tag="val"):
    probs = model.predict_proba(X)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    loss = log_loss(y, probs)
    return loss, acc

train_loss, train_acc = evaluate(pipeline, X_train, y_train, tag="train")
val_loss, val_acc = evaluate(pipeline, X_val, y_val, tag="val")

# Plot losses
losses = {
    "train": train_loss,
    "val": val_loss
}
accuracies = {
    "train": train_acc,
    "val": val_acc
}

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.barplot(x=list(losses.keys()), y=list(losses.values()))
plt.title("Loss Comparison")

plt.subplot(1, 2, 2)
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.title("Accuracy Comparison")
plt.tight_layout()
plt.savefig("loss_accuracy_plot.png")

# Test Set Evaluation
test_preds = pipeline.predict(X_test)
test_probs = pipeline.predict_proba(X_test)
test_loss = log_loss(y_test, test_probs)
test_acc = accuracy_score(y_test, test_preds)

print("\n=== Final Test Set Report ===")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_preds))

# Save model
joblib.dump(pipeline, "tfidf_heading_classifier.joblib")