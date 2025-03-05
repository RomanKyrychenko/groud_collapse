import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingClassifier
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

input_file = r"input/train.csv"
df = pd.read_csv(input_file).sample(frac=1, random_state=42)

X = df.iloc[:, :7]
scaler = StandardScaler()
scaler.fit(X)

# save scaler
scaler_filename = r'output/scaler.pkl'
joblib.dump(scaler, scaler_filename)

y = df.loc[:, 'collapse ']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

base_models = [('RandomForest', rf_model), ('NeuralNetwork', nn_model)]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=RandomForestClassifier(random_state=42, class_weight="balanced")
)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'final_estimator__n_estimators': [50, 100, 200],
    'final_estimator__max_depth': [2, 5],
    'final_estimator__min_samples_split': [2, 5, 10],
    'final_estimator__min_samples_leaf': [1, 2, 4],
    'RandomForest__n_estimators': [50, 100, 200],
    'RandomForest__max_depth': [2, 5],
    'NeuralNetwork__hidden_layer_sizes': [(100,), (100, 50)],
    'NeuralNetwork__max_iter': [1000, 2000]
}

# Use GridSearchCV for hyperparameter tuning with AUC as the scoring metric
grid_search = GridSearchCV(estimator=stacking_model, param_grid=param_grid,
                           scoring=make_scorer(roc_auc_score), cv=5, n_jobs=-1, verbose=3)

# Fit the model
grid_search.fit(scaler.transform(X), y)

# Get the best model
best_model = grid_search.best_estimator_

print(grid_search.best_params_)

# Perform cross-validation to evaluate the model based on AUC score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = scaler.transform(X.iloc[train_index]), scaler.transform(X.iloc[test_index])
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    best_model.fit(X_train, y_train)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(auc)

average_auc = sum(auc_scores) / len(auc_scores)
print(f'Average AUC: {average_auc:.2f}')

# Save the best model
model_filename = r'output/stacked_model.pkl'
joblib.dump(best_model, model_filename)

# Print feature importances from the final estimator
final_estimator = best_model.final_estimator_
if hasattr(final_estimator, 'feature_importances_'):
    feature_importances = final_estimator.feature_importances_
    for i, importance in enumerate(feature_importances):
        print(f'Feature {i}: {importance:.4f}')

print("Model saved")
