import pandas as pd
from sklearn.metrics import roc_auc_score

input_file = r"input/train.csv"
df = pd.read_csv(input_file).sample(frac=1, random_state=42)

test_file = r"input/test.csv"
test_df = pd.read_csv(test_file)

rs = test_df.merge(df.groupby(['the thickness of the surface fill layer',
       'distance to underground rivers and blind ditches',
       'burial depth of the top layer of saturated silty sand soil',
       'the density of the drainage pipe network',
       'burial depth of the underground confined water level', 'rainfall',
       'the thickness of the soft soil layer'], as_index=False).agg({'collapse ': 'mean'}), how='left', on=['the thickness of the surface fill layer',
       'distance to underground rivers and blind ditches',
       'burial depth of the top layer of saturated silty sand soil',
       'the density of the drainage pipe network',
       'burial depth of the underground confined water level', 'rainfall',
       'the thickness of the soft soil layer'], validate='many_to_one').fillna(0)

print("ROC-AUC score:", roc_auc_score(rs['collapse _x'], rs['collapse _y']))