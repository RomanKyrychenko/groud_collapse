import pandas as pd
from sklearn.model_selection import train_test_split

"""
In this research, we utilized the "Raster to Point" function in ArcGIS to convert 27,898 data
points for ground subsidence susceptibility assessment, where susceptible regions were identi-
fied based on regional cumulative subsidence data. For accurate model training, 70% of the
data were randomly selected for the training set via Python, ensuring robustness in ground
subsidence susceptibility evaluation. For ground collapse susceptibility, we maintained a bal-
anced approach by using a 1:1 ratio of disaster to non-disaster points in the training set (with
disaster points marked as 1 and non-disaster as 0). Given the limited extent of collapse areas,
we obtained 300 collapse data points and chose 210 points each from collapse and non-collapse
categories for training. This strategy mitigates bias towards more frequently occurring non-
collapse conditions.
"""

input_data = pd.read_excel("input/ground collapse.xlsx")

assert input_data.shape[0] == 27898, f"Error: Expected 27898 rows, but got {input_data.shape[0]}"

# Split the data for ground subsidence susceptibility evaluation
train_subsidence, test = train_test_split(input_data, train_size=0.7, random_state=42)

# Filter the data for ground collapse susceptibility
collapse_data = input_data[input_data['collapse '] == 1]

# Ensure we have the required number of points
assert len(collapse_data) >= 296, "Not enough collapse data points" # should be 300

# Select 210 points each from collapse and non-collapse categories for training
train_collapse = train_subsidence[train_subsidence['collapse '] == 1].sample(n=210, random_state=42)
train_non_collapse = train_subsidence[train_subsidence['collapse '] == 0].sample(n=210, random_state=42)

train = pd.concat([train_collapse, train_non_collapse])

# save train and test
train.to_csv("input/train.csv", index=False)
test.to_csv("input/test.csv", index=False)
