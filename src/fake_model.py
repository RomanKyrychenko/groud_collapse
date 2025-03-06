import pandas as pd
from sklearn.metrics import roc_auc_score


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

if __name__ == "__main__":
    input_file = r"../input/train.csv"
    test_file = r"../input/test.csv"
    model = FakeModel(input_file, test_file)
    print("ROC-AUC score:", model.merge_and_evaluate())