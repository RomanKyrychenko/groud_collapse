import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, input_file):
        self.input_data = pd.read_excel(input_file)
        self._validate_data()
        self.train = None
        self.test = None
        self.collapse_data = None
        self.train_subsidence = None

    def _validate_data(self):
        assert self.input_data.shape[0] == 27898, f"Error: Expected 27898 rows, but got {self.input_data.shape[0]}"
        print(f"Data contains {self.input_data.shape[0]} rows and {self.input_data.shape[1]} columns")
        print("Distribution of classes: ")
        print(self.input_data['collapse '].value_counts().to_dict())

    def split_data(self):
        self.train_subsidence, self.test = train_test_split(self.input_data, train_size=0.7, random_state=42)

    def filter_collapse_data(self):
        self.collapse_data = self.input_data[self.input_data['collapse '] == 1]
        assert len(self.collapse_data) >= 296, "Not enough collapse data points"

    def create_train_set(self):
        train_collapse = self.train_subsidence[self.train_subsidence['collapse '] == 1].sample(n=210, random_state=42)
        train_non_collapse = self.train_subsidence[self.train_subsidence['collapse '] == 0].sample(n=210, random_state=42)
        self.train = pd.concat([train_collapse, train_non_collapse])

    def save_data(self, train_file: str, test_file: str):
        """
        Saves the train and test datasets to specified file paths in CSV format.

        This method allows for persisting the train and test datasets into external
        CSV files. Each file is saved without including index data as part of the
        output.

        :param train_file: The file path where the train dataset will be saved.
        :type train_file: str
        :param test_file: The file path where the test dataset will be saved.
        :type test_file: str
        :return: None
        """
        self.train.to_csv(train_file, index=False)
        self.test.to_csv(test_file, index=False)

if __name__ == "__main__":
    preprocessor = DataPreprocessor("../input/ground collapse.xlsx")
    preprocessor.split_data()
    preprocessor.filter_collapse_data()
    preprocessor.create_train_set()
    preprocessor.save_data("input/train.csv", "input/test.csv")
