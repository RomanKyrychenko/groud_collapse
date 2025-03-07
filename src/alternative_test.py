import joblib
from src.original_test import ModelEvaluator


class AlternativeModelEvaluator(ModelEvaluator):
    def __init__(self, model_filename, test_file, scaler_filename):
        super().__init__(model_filename, test_file)
        self.scaler = joblib.load(scaler_filename)
        self.X_test = self.scaler.transform(self.X_test)
        #self.y_test = 1-self.y_test


if __name__ == "__main__":
    evaluator = AlternativeModelEvaluator(
        model_filename=r'../output/alternative_stacked_model.pkl',
        test_file=r"../input/alternative_test.csv",
        scaler_filename=r'../output/scaler.pkl'
    )
    evaluator.evaluate(roc_curve_filename=r'../output/alternative_roc_curve.png', result_excel_file=r'../output/alternative_predicted_results.csv')
