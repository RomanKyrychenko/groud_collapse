import pandas as pd
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

test_file = r"input/test.csv"
test_df = pd.read_csv(test_file)

model_filename = r'output/stacked_model.pkl'
stacking_model = joblib.load(model_filename)

X_test = test_df.iloc[:, :7]

y_test = test_df.iloc[:, 7]

positive_class_label = y_test.max()
y_test_binary = label_binarize(y_test, classes=[0, positive_class_label]).ravel()

y_score = stacking_model.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test_binary, y_score[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROCcurve (s = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('false')
plt.ylabel('true')
plt.title('curve')
plt.legend(loc="lower right")

roc_curve_filename = r'output/roc_curve nerve.png'
plt.savefig(roc_curve_filename)

test_df['results'] = stacking_model.predict(X_test)
result_excel_file = r'output/predicted_results.csv'
test_df.to_csv(result_excel_file, index=False)

print("saved2")