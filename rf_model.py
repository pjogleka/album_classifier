import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

# Load dataframe from csv
lyrics_array = np.loadtxt('data.csv', delimiter=",", dtype=str)
lyrics_df = pd.DataFrame(lyrics_array, columns=['Album', 'Song', 'Lyric'])

X = lyrics_df['Lyric']
y = lyrics_df['Album']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_df=0.7)),
    ("clf", RandomForestClassifier(random_state=42))
    ])

param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [10, 20, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", None]
    }

# Test models from param_grid
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc_ovr", n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best ROC AUC CV score: {grid.best_score_}")

y_pred = grid.predict(X_test)

# Binarize labels for multiclass ROC AUC
classes = grid.best_estimator_.named_steps["clf"].classes_
y_test_bin = label_binarize(y_test, classes=classes)

# Predicted probabilities (n_samples × n_classes)
y_proba = grid.predict_proba(X_test)

# Calculate accuracy and ROC AUC score
acc_score = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
print(f"ROC AUC score: {roc_auc}")

# Get classification report
print(classification_report(y_test, y_pred, zero_division=0))

# Save best model
model = grid.best_estimator_
joblib.dump(model, "best_model.joblib")

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
class_labels = model.named_steps["clf"].classes_

# Convert to abbreviated names
album_abbr = [abbr_labels[album] for album in class_labels]

# Save matrix
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=album_abbr, cmap="Blues", ax=ax, xticks_rotation=45)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Random Forest Confusion Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("plots/rf_confusion_matrix.png")
plt.close()

# Binarize labels for multiclass ROC AUC
classes = model.named_steps["clf"].classes_
y_test_bin = label_binarize(y_test, classes=classes)

# Predicted probabilities (n_samples × n_classes)
y_proba = model.predict_proba(X_test)

# Get ROC curves
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC AUC Curve")
plt.legend(loc="lower right", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/rf_roc_auc.png")
plt.close()
