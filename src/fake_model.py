import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


class FakeModel:
    def __init__(self, input_file, test_file):
        self.input_file = input_file
        self.test_file = test_file
        self.df = pd.read_csv(input_file).sample(frac=1, random_state=42)
        self.test_df = pd.read_csv(test_file)

    def merge_and_evaluate(self):
        rs = self.test_df.merge(
            self.df.groupby([
                'the thickness of the surface fill layer',
                'distance to underground rivers and blind ditches',
                'burial depth of the top layer of saturated silty sand soil',
                'the density of the drainage pipe network',
                'burial depth of the underground confined water level',
                'rainfall',
                'the thickness of the soft soil layer'
            ], as_index=False).agg({'collapse ': 'mean'}),
            how='left',
            on=[
                'the thickness of the surface fill layer',
                'distance to underground rivers and blind ditches',
                'burial depth of the top layer of saturated silty sand soil',
                'the density of the drainage pipe network',
                'burial depth of the underground confined water level',
                'rainfall',
                'the thickness of the soft soil layer'
            ],
            validate='many_to_one'
        ).fillna(0)
        return roc_auc_score(rs['collapse _x'], rs['collapse _y'])

    def plot_roc_curve(self, roc_curve_filename, title: str = 'Receiver Operating Characteristic'):
        # Merge data to get predictions like in merge_and_evaluate
        merged_df = self.test_df.merge(
            self.df.groupby([
                'the thickness of the surface fill layer',
                'distance to underground rivers and blind ditches',
                'burial depth of the top layer of saturated silty sand soil',
                'the density of the drainage pipe network',
                'burial depth of the underground confined water level',
                'rainfall',
                'the thickness of the soft soil layer'
            ], as_index=False).agg({'collapse ': 'mean'}),
            how='left',
            on=[
                'the thickness of the surface fill layer',
                'distance to underground rivers and blind ditches',
                'burial depth of the top layer of saturated silty sand soil',
                'the density of the drainage pipe network',
                'burial depth of the underground confined water level',
                'rainfall',
                'the thickness of the soft soil layer'
            ],
            validate='many_to_one'
        ).fillna(0)

        fpr, tpr, _ = roc_curve(merged_df['collapse _x'], merged_df['collapse _y'])
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

    def save_predictions(self, result_csv_file):
        merged_df = self.test_df.merge(
            self.df.groupby([
                'the thickness of the surface fill layer',
                'distance to underground rivers and blind ditches',
                'burial depth of the top layer of saturated silty sand soil',
                'the density of the drainage pipe network',
                'burial depth of the underground confined water level',
                'rainfall',
                'the thickness of the soft soil layer'
            ], as_index=False).agg({'collapse ': 'mean'}),
            how='left',
            on=[
                'the thickness of the surface fill layer',
                'distance to underground rivers and blind ditches',
                'burial depth of the top layer of saturated silty sand soil',
                'the density of the drainage pipe network',
                'burial depth of the underground confined water level',
                'rainfall',
                'the thickness of the soft soil layer'
            ],
            validate='many_to_one'
        ).fillna(0)
        merged_df['results'] = merged_df['collapse _y']
        merged_df.to_csv(result_csv_file, index=False)


if __name__ == "__main__":
    input_file = r"../input/train.csv"
    test_file = r"../input/test.csv"
    model = FakeModel(input_file, test_file)
    print("ROC-AUC score:", model.merge_and_evaluate())
    model.plot_roc_curve("fake_model_roc_curve.png")
    model.save_predictions("fake_model_results.csv")