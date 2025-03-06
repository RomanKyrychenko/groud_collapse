from sklearn.model_selection import train_test_split
from src.data_preproc import DataPreprocessor


class AlternativeDataPreprocessor(DataPreprocessor):
    def __init__(self, input_file):
        super().__init__(input_file)
        self.input_data = self.input_data.drop_duplicates()
        self._validate_data()

    def _validate_data(self):
        print(f"Data contains {self.input_data.shape[0]} rows and {self.input_data.shape[1]} columns")
        print("Distribution of classes: ")
        print(self.input_data['collapse '].value_counts())

    def split_data(self):
        self.train, self.test = train_test_split(self.input_data, train_size=0.7, random_state=42)

    def save_data(self, train_file, test_file):
        self.train.to_csv(train_file, index=False)
        self.test.to_csv(test_file, index=False)


if __name__ == "__main__":
    preprocessor = AlternativeDataPreprocessor("../input/ground collapse.xlsx")
    preprocessor.split_data()
    preprocessor.save_data("input/train.csv", "input/test.csv")