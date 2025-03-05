import pandas as pd
from sklearn.model_selection import train_test_split

input_data = pd.read_excel("input/ground collapse.xlsx")

assert input_data.shape[0] == 27898, f"Error: Expected 27898 rows, but got {input_data.shape[0]}"

print(input_data.drop_duplicates()['collapse '].value_counts())

input_data = input_data.drop_duplicates()

# Split the data for ground subsidence susceptibility evaluation
train, test = train_test_split(input_data, train_size=0.7, random_state=42)

# save train and test
train.to_csv("input/train.csv", index=False)
test.to_csv("input/test.csv", index=False)
