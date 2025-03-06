import pandas as pd
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    def __init__(self, model_filename, test_file):
        self.model_filename = model_filename
        self.test_file = test_file
        self.test_df = pd.read_csv(test_file)
        self.stacking_model = joblib.load(model_filename)
        self.X_test = self.test_df.iloc[:, :7]
        self.y_test = self.test_df.iloc[:, 7]
        self.positive_class_label = self.y_test.max()
        self.y_test_binary = label_binarize(self.y_test, classes=[0, self.positive_class_label]).ravel()

    def plot_roc_curve(self, roc_curve_filename, title: str = 'Receiver Operating Characteristic'):
        y_score = self.stacking_model.predict_proba(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test_binary, y_score[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(roc_curve_filename)

    def save_predictions(self, result_excel_file):
        self.test_df['results'] = self.stacking_model.predict(self.X_test)
        self.test_df['results_proba'] = self.stacking_model.predict_proba(self.X_test)[:, 1]
        self.test_df.to_csv(result_excel_file, index=False)

    def evaluate(self, roc_curve_filename, result_excel_file):
        self.plot_roc_curve(roc_curve_filename)
        self.save_predictions(result_excel_file)
        print("Evaluation completed and results saved.")


if __name__ == "__main__":
    evaluator = ModelEvaluator(model_filename=r'../output/stacked_model.pkl', test_file=r"../input/test.csv")
    evaluator.evaluate(roc_curve_filename=r'output/roc_curve.png', result_excel_file=r'../output/predicted_results.csv')
