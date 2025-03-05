import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingClassifier
import joblib

input_file = r"input/train.csv"
df = pd.read_csv(input_file)

X = df.iloc[:, :7]  #
y = df.iloc[:, 7]  #

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

base_models = [('RandomForest', rf_model), ('NeuralNetwork', nn_model)]

stacking_model = StackingClassifier(estimators=base_models,
                                    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))

kf = KFold(n_splits=5)
accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

average_accuracy = sum(accuracies) / len(accuracies)
print(f':average_accuracy {average_accuracy:.2f}')

model_filename = r'output/stacked_model.pkl'
joblib.dump(stacking_model, model_filename)

print("saved")

