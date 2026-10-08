"""Reproduce a held-out baseline; preprocessing is fitted on training data only."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent

def main():
    source = ROOT / 'titanic/train.csv'
    data = pd.read_csv(source)
    numerical = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    categorical = ['Sex', 'Embarked']
    X, y = data[numerical + categorical], data['Survived']
    assert y.notna().all() and set(y.unique()) == {0, 1}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42)
    preprocessing = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numerical),
        ('categorical', Pipeline([
            ('missing', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical),
    ])
    model = Pipeline([
        ('prepare', preprocessing),
        ('classifier', RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = {
        'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows': len(data), 'training_rows': len(X_train), 'test_rows': len(X_test),
        'split': 'stratified 80/20, random_state=42',
        'features': numerical + categorical, 'sklearn_version': sklearn.__version__,
        'accuracy': round(accuracy_score(y_test, predictions), 6),
        'f1': round(f1_score(y_test, predictions), 6),
        'roc_auc': round(roc_auc_score(y_test, probabilities), 6),
        'majority_baseline_accuracy': round(accuracy_score(y_test, baseline.predict(X_test)), 6),
        'confusion_matrix': matrix.tolist(),
        'limitations': 'One held-out split; educational baseline, not a competition score or externally validated model. Separate from the dashboard cached model.',
    }
    out = ROOT / 'docs/results'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.DataFrame(matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']).to_csv(out / 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.imshow(matrix, cmap='Blues')
    for (row, col), value in __import__('numpy').ndenumerate(matrix):
        ax.text(col, row, str(value), ha='center', va='center', fontsize=22)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Did not survive', 'Survived'],
           yticklabels=['Did not survive', 'Survived'], xlabel='Predicted', ylabel='Actual',
           title=f'Titanic: stratified held-out evaluation ({len(X_test)} passengers)')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
