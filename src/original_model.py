import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import joblib


class StackingModel:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_csv(input_file)
        self.df = self.df.sample(frac=1, random_state=42)
        self.X = self.df.iloc[:, :7]
        self.y = self.df.loc[:, 'collapse ']
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        self.stacking_model = StackingClassifier(
            estimators=[('RandomForest', self.rf_model), ('NeuralNetwork', self.nn_model)],
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.kf = KFold(n_splits=5)
        self.accuracies = []

    def train_and_evaluate(self):
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.stacking_model.fit(X_train, y_train)
            y_pred = self.stacking_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.accuracies.append(accuracy)
        average_accuracy = sum(self.accuracies) / len(self.accuracies)
        print(f'average_accuracy {average_accuracy:.2f}')

    def save_model(self, model_filename):
        joblib.dump(self.stacking_model, model_filename)
        print("saved")


if __name__ == "__main__":
    model = StackingModel(input_file=r"../input/train.csv")
    model.train_and_evaluate()
    model.save_model(model_filename=r'../output/stacked_model.pkl')
