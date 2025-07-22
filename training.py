import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.utils import shuffle, compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load CSV
df = pd.read_csv("data.csv")
df = df.dropna(subset=['text', 'label'])

# Train/Val/Test Split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

X_train_raw, y_train = train_df['text'].tolist(), train_df['label'].tolist()
X_val_raw, y_val = val_df['text'].tolist(), val_df['label'].tolist()
X_test_raw, y_test = test_df['text'].tolist(), test_df['label'].tolist()

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_raw)
X_val = vectorizer.transform(X_val_raw)
X_test = vectorizer.transform(X_test_raw)

# Class labels
classes = np.unique(y_train)

# Manually compute class weights
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights_array))

# Classifier: SGDClassifier as Logistic Regression
clf = SGDClassifier(
    loss='log_loss',
    max_iter=1,
    warm_start=True,
    random_state=42
)

# Training loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
epochs = 900

for epoch in range(epochs):
    # Shuffle training data
    X_train, y_train = shuffle(X_train, y_train, random_state=epoch)
    
    # Compute sample weights
    sample_weight = np.array([class_weight_dict[label] for label in y_train])

    # Partial fit with sample weights
    clf.partial_fit(X_train, y_train, classes=classes, sample_weight=sample_weight)

    # Predictions
    train_probs = clf.predict_proba(X_train)
    val_probs = clf.predict_proba(X_val)
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)

    # Metrics
    train_loss = log_loss(y_train, train_probs)
    val_loss = log_loss(y_val, val_probs)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Plotting loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_accuracies, label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_progress.png")
plt.show()

# Final Test Evaluation
test_preds = clf.predict(X_test)
test_probs = clf.predict_proba(X_test)
test_loss = log_loss(y_test, test_probs)
test_acc = accuracy_score(y_test, test_preds)

print("\n=== Final Test Set Report ===")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_preds))

# Save model
joblib.dump((vectorizer, clf), "tfidf_sgd_heading_classifier.joblib")