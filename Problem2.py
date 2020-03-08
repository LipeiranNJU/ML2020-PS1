print("TEST")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
dataset = pd.read_csv("./data.csv")
print(dataset['output'][0])
dataset['Prediction'] = None
dataset['Type'] = None
P = list()
R = list()
TP = FP = TN = FN = 0
for i in range(len(dataset)):
    bar = dataset.loc[i, 'output']
    for j in range(len(dataset)):
        if dataset['output'][j] <= bar:
            dataset.loc[j, 'Prediction'] = 0
            if dataset.loc[j, 'Prediction'] == dataset.loc[j, 'label']:
                dataset.loc[j, 'Type'] = 'TN'
                TN += 1
            else:
                dataset.loc[j, 'Type'] = 'FN'
                FN += 1
        else:
            dataset.loc[j, 'Prediction'] = 1
            if dataset.loc[j, 'Prediction'] == dataset.loc[j, 'label']:
                dataset.loc[j, 'Type'] = 'TP'
                TP += 1
            else:
                dataset.loc[j, 'Type'] = 'FP'
                FP += 1
    P.append(TP / (TP + FP))
    R.append(TP / (TP + FN))
    if (i + 1) % 50 == 0:
        print("Having test " + str(i + 1) + " times.")


plt.plot(R, P, 'k')

plt.title('Receiver Operating Characteristic')
plt.plot([(0, 0), (1, 1)], 'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 01.01])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# from sklearn.metrics import precision_recall_curve
#
# plt.figure("P-R Curve")
# plt.title('Precision/Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# y_true = np.array(dataset['label'])
# y_scores = np.array(dataset['output'])
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
# # print(precision)
# # print(recall)
# # print(thresholds)
# plt.plot(recall, precision)
# plt.show()
