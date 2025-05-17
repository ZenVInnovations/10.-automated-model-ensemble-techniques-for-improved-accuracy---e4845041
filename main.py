
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC

# Load dataset
df = sns.load_dataset('titanic')

# Display first few rows
print(df.head())

# Drop irrelevant/unnecessary columns
df = df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'])

# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and label
X = df.drop('survived', axis=1)
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define base models
models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
]
voting_clf = VotingClassifier(estimators=models, voting='soft')
voting_clf.fit(X_train, y_train)
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
bagging_clf.fit(X_train, y_train)
boosting_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
boosting_clf.fit(X_train, y_train)
stacking_clf = StackingClassifier(
    estimators=models,
    final_estimator=SVC(probability=True)
)
stacking_clf.fit(X_train, y_train)
# Store models in dictionary
ensemble_models = {
    'Voting': voting_clf,
    'Bagging': bagging_clf,
    'Boosting': boosting_clf,
    'Stacking': stacking_clf
}

# Evaluate each
for name, model in ensemble_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
best_model_name = max(ensemble_models, key=lambda name: accuracy_score(y_test, ensemble_models[name].predict(X_test)))
best_model_acc = accuracy_score(y_test, ensemble_models[best_model_name].predict(X_test))

print(f"\n Best Ensemble Model: {best_model_name} with Accuracy = {best_model_acc:.4f}")
accuracies = [accuracy_score(y_test, model.predict(X_test)) for model in ensemble_models.values()]
plt.bar(ensemble_models.keys(), accuracies, color='skyblue')
plt.ylabel("Accuracy")
plt.title("Ensemble Model Comparison")
plt.ylim(0, 1)
plt.show()




