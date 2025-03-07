from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from src.original_model import StackingModel
from sklearn.ensemble import RandomForestClassifier, StackingClassifier


class AlternativeStackingModel(StackingModel):
    def __init__(self, input_file):
        super().__init__(input_file)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.stacking_model = StackingClassifier(
            estimators=[('RandomForest', self.rf_model), ('NeuralNetwork', self.nn_model)],
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        )
        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)
        self.y = 1-self.y
        self.param_grid = {
            'final_estimator__n_estimators': [10],
            'final_estimator__max_depth': [2, 5],
            #'final_estimator__min_samples_split': [2, 5],
            #'final_estimator__min_samples_leaf': [1, 2],
            'RandomForest__n_estimators': [10],
            'RandomForest__max_depth': [2, 5],
            'NeuralNetwork__hidden_layer_sizes': [(100,), (100, 50)],
            'NeuralNetwork__max_iter': [2000]
        }
        self.best_model = None

    def save_scaler(self, scaler_filename):
        joblib.dump(self.scaler, scaler_filename)

    def tune_hyperparameters(self):
        grid_search = GridSearchCV(estimator=self.stacking_model, param_grid=self.param_grid,
                                   scoring=make_scorer(roc_auc_score), cv=3, n_jobs=-1, verbose=3)
        grid_search.fit(self.X, self.y)
        self.stacking_model = grid_search.best_estimator_
        print(grid_search.best_params_)

    def train_and_evaluate(self):
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index, :], self.X[test_index, :]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.stacking_model.fit(X_train, y_train)
            y_pred = self.stacking_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.accuracies.append(accuracy)
        average_accuracy = sum(self.accuracies) / len(self.accuracies)
        print(f'average_accuracy {average_accuracy:.2f}')

    def print_feature_importances(self):
        final_estimator = self.stacking_model.final_estimator_
        if hasattr(final_estimator, 'feature_importances_'):
            feature_importances = final_estimator.feature_importances_
            for i, importance in enumerate(feature_importances):
                print(f'Feature {i}: {importance:.4f}')


if __name__ == "__main__":
    alternative_model = AlternativeStackingModel(input_file=r"../input/alternative_train.csv")
    alternative_model.save_scaler(scaler_filename=r'../output/scaler.pkl')
    alternative_model.tune_hyperparameters()
    alternative_model.train_and_evaluate()
    alternative_model.save_model(model_filename=r'../output/alternative_stacked_model.pkl')
    alternative_model.print_feature_importances()

