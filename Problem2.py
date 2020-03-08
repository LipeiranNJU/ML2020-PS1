from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv("./data.csv")
print(dataset)
Pre = list()
Rec = list()
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
    if i % 50 == 49:
        print(str((i + 1) / len(dataset) * 100) + "% have tested.")

Dots = list()


def TakeRec(item):
    return item[0]


for dot in zip(Rec, Pre):
    Dots.append(dot)
Dots.sort(key=TakeRec)
Rec = [Dots[i][0] for i in range(len(Dots))]
Pre = [Dots[i][1] for i in range(len(Dots))]
plt.plot(Rec, Pre)
plt.show()
