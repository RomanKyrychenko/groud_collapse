import joblib
import pandas as pd
from src.original_test import ModelEvaluator


class AlternativeModelEvaluator(ModelEvaluator):
    def __init__(self, model_filename, test_file, scaler_filename):
        super().__init__(model_filename, test_file)
        self.scaler = joblib.load(scaler_filename)
        names = self.X_test.columns
        self.X_test = self.scaler.transform(self.X_test)
        self.X_test = pd.DataFrame(self.X_test, columns=names)


if __name__ == "__main__":
    evaluator = AlternativeModelEvaluator(
        model_filename=r'../output/stacked_model.pkl',
        test_file=r"../input/test.csv",
        scaler_filename=r'../output/scaler.pkl'
    )
    evaluator.evaluate(roc_curve_filename=r'output/roc_curve.png', result_excel_file=r'../output/predicted_results.csv')
