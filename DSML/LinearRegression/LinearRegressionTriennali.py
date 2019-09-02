"""
Algoritmo di linear regression, molto semplice ed intuitivo.
Vengono dati in input i campi "Tipo_Mat" e "Voto_Diploma" (letti dal file ListStudent.txt)
e si cerca di estrapolare un'equazione in grado di predirre il numero di CFU fatti al primo anno.
I risultati sono pessimi poichè:
1. La quantità di dati a disposizione è minima
2. Gli attributi predittivi presi in esame sono altamente scorrelati fra di loro



A questo link è presente una spiegazione dettagliata di tutte le metriche di valutazione per gli algoritmi di regressione. https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import json
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/ListStudent.txt")) #ATTENZIONE AL PATH
Train_size = int((len(Maturità) / 100) * 80)
Test_size = int((len(Maturità)/100) * 20)
Train_set = []
Test_set = []
Result_Test = []

for i in range(0, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TrainTemp = [tipo_maturità, voto_diploma]
    Result.append(int(Maturità[i][0][2]))
    Train_set.append(TrainTemp)

Train_set, Test_set, Result,  Result_Test = train_test_split(Train_set, Result, test_size=0.3)
Train_set, Result = np.array(Train_set), np.array(Result)
Train_set, Result = np.array(Train_set), np.array(Result)
model = LinearRegression()
model.fit(Train_set, Result)
r_sq = model.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", model.intercept_)
print("slope: ", model.coef_)
maximum = 0.0
sw = 0
minus = 0.0
y_pred_pred = []
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    predictiveTemp = [[tipo_maturità, voto_diploma]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = model.predict(predictiveTemp)
    y_pred = float(y_pred)
    y_pred_pred.append(y_pred)
    if y_pred > 0:
        if sw == 0:
            minus = y_pred
            min_school = tipo_maturità
            sw = 1
        elif minus > y_pred:
            minus = y_pred
            min_school = tipo_maturità
        if y_pred > maximum:
            maximum = y_pred
            max_school = tipo_maturità

print("Highest score: ", maximum, "of type ", sep="\n")
print("Worst score: ", minus, "of school ", sep="\n")


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("R2Score : ", r2_score(Result_Test, y_pred_pred))
print("MSE : ", mean_squared_error(Result_Test, y_pred_pred))

predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set = []
test_set = []
train_result = []
test_result = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10],
                          predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][17],
                          predictiveAttributeNotDegree[i][18]])
        train_result.append([predictiveAttributeDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10],
                          predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][17],
                          predictiveAttributeNotDegree[i][18]])
        test_result.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10],
                          predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][17],
                          predictiveAttributeNotDegree[i][18]])
        train_result.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10],
                          predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][17],
                          predictiveAttributeNotDegree[i][18]])
        test_result.append([predictiveAttributeNotDegree[i][2]])

lin_reg_tot = LinearRegression()
lin_reg_tot.fit(train_set,train_result)
r_sq = lin_reg_tot.score(train_set, train_result)
print("-----ALL ATTRIBUTE-----: Coefficient of determination: ", r_sq)
print("-----ALL ATTRIBUTE-----: slope: ", lin_reg_tot.coef_)