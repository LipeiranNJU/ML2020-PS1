from matplotlib import pyplot as plt
import pandas as pd

dataset = pd.read_csv("./data.csv")
Pre = list()
Rec = list()
TPR = list()
FPR = list()
for i in range(len(dataset)):
    bar = dataset.loc[i, 'output']
    dataset['Prediction'] = dataset.apply(lambda x: 1 if x['output'] >= bar else 0, axis=1)
    dataset['Type'] = dataset.apply(lambda x: True if x['label'] == x['Prediction'] else False, axis=1)
    dataset['TP'] = dataset.apply(lambda x: 1 if x['Type'] is True and x['Prediction'] == 1 else 0, axis=1)
    dataset['FP'] = dataset.apply(lambda x: 1 if x['Type'] is False and x['Prediction'] == 1 else 0, axis=1)
    dataset['TN'] = dataset.apply(lambda x: 1 if x['Type'] is True and x['Prediction'] == 0 else 0, axis=1)
    dataset['FN'] = dataset.apply(lambda x: 1 if x['Type'] is False and x['Prediction'] == 0 else 0, axis=1)

    TP = sum(dataset['TP'])
    FP = sum(dataset['FP'])
    TN = sum(dataset['TN'])
    FN = sum(dataset['FN'])

    Pre.append(TP / (TP + FP))
    Rec.append(TP / (TP + FN))
    TPR.append(TP / (TP + FN))
    FPR.append(FP / (TN + FP))
    if i % 50 == 49:
        print(str((i + 1) / len(dataset) * 100) + "% have tested.")

Dots1 = list()
Dots2 = list()


def TakeRec(item):
    return item[0]


def TakeFPR(item):
    return item[0]


for dot in zip(Rec, Pre):
    Dots1.append(dot)

for dot in zip(FPR, TPR):
    Dots2.append(dot)
Dots1.sort(key=TakeRec)
Dots2.sort(key=TakeFPR)
Rec = [Dots1[i][0] for i in range(len(Dots1))]
Pre = [Dots1[i][1] for i in range(len(Dots1))]
FPR = [Dots2[i][0] for i in range(len(Dots2))]
TPR = [Dots2[i][1] for i in range(len(Dots2))]

AUC = 0
for i in range(len(FPR) - 1):
    AUC += 1 / 2 * (FPR[i+1] - FPR[i]) * (TPR[i] + TPR[i + 1])

print("AUC = {}".format(AUC))
plt.figure(0)
plt.plot(Rec, Pre)
plt.xlabel("Rec")
plt.ylabel("Pre")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("P-R curve")
plt.figure(1)
plt.plot(FPR, TPR)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("ROC")
plt.fill_between(FPR, 0, TPR, color="green", alpha=0.25)
plt.annotate("AUC = {}".format(AUC), xy=(0.2, 0.2), xytext=(0.4, 0.2),arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
