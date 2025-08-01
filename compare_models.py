import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import joblib

# Load dataframe from csv
lyrics_array = np.loadtxt('data.csv', delimiter=",", dtype=str)
lyrics_df = pd.DataFrame(lyrics_array, columns=['Album', 'Song', 'Lyric'])

X = lyrics_df['Lyric']
y = lyrics_df['Album']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary of models and parameters for grid search
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Multinomial NB": MultinomialNB(),
    "SGD Classifier": OneVsRestClassifier(SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)),
    }
param_grids = {
    "Logistic Regression": {
        "clf__C": [0.1, 1, 10],
        },
    "Random Forest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20]
        },
    "Multinomial NB": {
        "clf__alpha": [0.5, 1.0, 1.5]
        },
    "SGD Classifier": {
        "clf__estimator__alpha": [1e-4, 1e-3],
        "clf__estimator__penalty": ["l2", "elasticnet"]
        }
    }

# Binarize labels for multiclass ROC AUC
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

# Vectorize data with TF-IDF

# Initialize score arrays for plotting
acc_scores = np.zeros(len(models))
roc_auc_scores = np.zeros(len(models))
cv_scores = np.zeros(len(models))
grids = {}

# Loop through models for grid search
for i, (clf_name, clf) in enumerate(models.items()):
    
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_df=0.7, max_features=5000)),
        ("clf", clf)
        ])
    
    grid = GridSearchCV(pipe, param_grids[clf_name], cv=5, scoring="accuracy", n_jobs=1, verbose=0)
    grid.fit(X_train, y_train)
    
    grids[i] = grid.best_estimator_
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best ROC AUC CV score: {grid.best_score_}")
    cv_scores[i] = grid.best_score_
    
    y_pred = grid.predict(X_test)
    
    if hasattr(grid.best_estimator_["clf"], "predict_proba"):
        y_proba = grid.predict_proba(X_test)
    else:
        y_raw = grid.decision_function(X_test)
        y_proba = softmax(y_raw, axis=1)

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
        roc_auc_scores[i] = roc_auc
        print(f"ROC AUC score: {roc_auc}")
    
    acc_score = accuracy_score(y_test, y_pred)
    acc_scores[i] = acc_score
    print(f"Accuracy score: {acc_score}")
    print(classification_report(y_test, y_pred, zero_division=0))

best_grid = np.argmax(acc_scores)
joblib.dump(grids[best_grid], "best_model.joblib")

# Grouped bar chart using plt docs example
model_names = list(models.keys())
model_scores = {
    "Accuracy": np.around(acc_scores, decimals=2),
    "ROC AUC": np.around(roc_auc_scores, decimals=2),
    "CV": np.around(cv_scores, decimals=2)
    }

x = np.arange(len(model_names))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for model, score in model_scores.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, score, width, label=model)
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width, model_names)
ax.set_ylim(0, 1)
ax.legend(loc="upper right")

plt.savefig("plots/model_comp.png", dpi=300)
plt.show()