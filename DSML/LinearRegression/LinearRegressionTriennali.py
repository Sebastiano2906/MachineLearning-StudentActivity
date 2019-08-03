"""
Algoritmo di linear regression, molto semplice ed intuitivo.
Vengono dati in input i campi "Tipo_Mat" e "Voto_Diploma" (letti dal file ListStudent.txt)
e si cerca di estrapolare un'equazione in grado di predirre il numero di CFU fatti al primo anno.
I risultati sono pessimi poichè:
1. La quantità di dati a disposizione è minima
2. Gli attributi predittivi presi in esame sono altamente scorrelati fra di loro
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
Total_Set = []
Tota_Result = []
# Total_Write = []
# Total_Temp = []
for i in range(0, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TrainTemp = [tipo_maturità, voto_diploma]
    Total_Temp = [tipo_maturità, voto_diploma, int(Maturità[i][0][2])]
    Tota_Result.append(int(Maturità[i][0][2]))
    Result.append(int(Maturità[i][0][2]))
    Total_Set.append(TrainTemp)
    Train_set.append(TrainTemp)
    #Total_Write.append(Total_Temp)


# df = pd.DataFrame(data={"Tipo_Maturita, Voto_Diploma, CFU_Primo": Total_Write})
# df.to_csv("./TotalStudent.csv", sep=',', index=False,)
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


plot_learning_curves(model, Total_Set, Tota_Result)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("R2Score : ", r2_score(Result_Test, y_pred_pred))
print("MSE : ", mean_squared_error(Result_Test, y_pred_pred))