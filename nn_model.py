import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from nn import LyricsClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


# Load dataframe from csv
lyrics_array = np.loadtxt('data.csv', delimiter=",", dtype=str)
lyrics_df = pd.DataFrame(lyrics_array, columns=['Album', 'Song', 'Lyric'])

X = lyrics_df['Lyric']
y = lyrics_df['Album']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7, max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Normalize input
scaler = MaxAbsScaler()
X_train_tfidf = scaler.fit_transform(X_train_tfidf)
X_test_tfidf = scaler.transform(X_test_tfidf)

# Encode labels as integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

# Data loading
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the neural network
model = LyricsClassifier(input_dim=X_train_tensor.shape[1], hidden_dim=512, output_dim=len(le.classes_))

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# Evaluate
model.eval()
correct = 0
total = 0
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

y_test_bin = label_binarize(all_labels, classes=range(len(le.classes_)))

roc_auc = roc_auc_score(y_test_bin, all_probs, multi_class="ovr")

print(f"PyTorch NN Accuracy: {correct / total:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Album abbreviations for matrix
abbr_labels = {
    "Taylor Swift": "Debut",
    "Fearless (Taylor’s Version)": "Fearless",
    "Speak Now (Taylor’s Version)": "Speak Now",
    "Red (Taylor’s Version)": "Red",
    "1989 (Taylor’s Version)": "1989",
    "reputation": "reputation",
    "Lover": "Lover",
    "folklore (deluxe version)": "folklore",
    "evermore (deluxe version)": "evermore",
    "Midnights (The Til Dawn Edition)": "Midnights",
    "THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY": "TTPD"
    }

# Get class labels in the model
class_labels = le.classes_

# Convert to abbreviated names
album_abbr = [abbr_labels[album] for album in class_labels]

# Confusion matrix
y_pred = all_probs.argmax(axis=1)
cm = confusion_matrix(all_labels, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=album_abbr)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Neural Network Confusion Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("plots/nn_confusion_matrix.png", dpi=300)
plt.show()

# ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
for i, class_name in enumerate(le.classes_):
    RocCurveDisplay.from_predictions(y_test_bin[:, i], all_probs[:, i], ax=ax, name=class_name)
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("NN ROC AUC Curves")
plt.legend(loc="lower right", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/nn_roc_auc_curve.png", dpi=300)
plt.show()