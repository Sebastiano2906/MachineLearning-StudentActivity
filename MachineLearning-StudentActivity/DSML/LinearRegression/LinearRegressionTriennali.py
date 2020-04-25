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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score

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
    Result.append([int(Maturità[i][0][2])])
    Train_set.append(TrainTemp)

Train_set, Test_set, Result,  Result_Test = train_test_split(Train_set, Result, test_size=0.3)
Train_set, Result = np.array(Train_set), np.array(Result)
Train_set, Result = np.array(Train_set), np.array(Result)
model = LinearRegression()
model.fit(Train_set, Result)
pred1 = model.predict(Test_set)
newStudent = [[9,92]]
realValue = [42]
predicted = model.predict(newStudent)
prediction = []
for item in Test_set:
    items = [[item[0], item[1]]]
    prediction.append(model.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0][0]
print("Predetto : {}".format(predicted))
print("errore : {}".format(mean_squared_error(realValue,predicted)))
print(("MSE: {}".format(mean_squared_error(pred, Result_Test))))
r_sq = r2_score(Result_Test, pred1)
print("Coefficient of determination: ", r_sq)
print("intercept: ", model.intercept_)
print("slope: ", model.coef_)

import math
def metrics(m,X,y):
    yhat = m.predict(X)
    SS_Residual = sum((y-yhat)**2)
    SS_Total = sum((y-np.mean(y))**2)
    Scarto_totale = sum(abs(yhat-y))
    Scarto_totale_quadratico = sum((yhat-y)**2)
    mae = Scarto_totale / len(y)
    mse = Scarto_totale_quadratico / len(y)
    rmse = math.sqrt(mse)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adj_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-2 -1)
    return r_squared, adj_r_squared, mae, mse, rmse

r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(model, Test_set, Result_Test)
print("ADJ_R2Score: ", adj_r_squared)
print("MSE : ",MSE)
"""
plt.scatter(Test_set, Result_Test, color='black')
plt.plot(Test_set, pred1, color='blue', linewidth=3)
plt.show()"""
"""
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
plot_learning_curves(model,Train_set,Result)
theta_0 = model.intercept_
theta_1_500 = model.coef_

plt.figure()
plt.plot(Train_set,Result,'x')
theta_1 = theta_1_500
x = [[69,100]]
plt.plot(x,theta_0+x *theta_1,'r')
plt.show()

"""

#   QUI COMINCIA LA PROVA CON TUTTI GLI ATTRIBUTI


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
        train_set.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        train_result.append([predictiveAttributeDegree[i][2]])
    else:
        test_set.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        test_result.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8], predictiveAttributeNotDegree[i][9],
                          predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][17]])
        train_result.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8], predictiveAttributeNotDegree[i][9],
                          predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][17]])
        test_result.append([predictiveAttributeNotDegree[i][2]])

newStudent = [[633, 1355, 1, 1, 2013, 3, 2013, 92, 54, 9, 0]]
realValue = [42]
lin_reg_tot = LinearRegression()
lin_reg_tot.fit(train_set, train_result)
pred = lin_reg_tot.predict(test_set)
r_sq = r2_score(test_result, pred)
predicted = lin_reg_tot.predict(newStudent)
print("Predetto : {}".format(predicted))
print("errore : {}".format(mean_squared_error(realValue,predicted)))
prediction = []
for item in test_set:
    items = [[item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10]]]
    prediction.append(lin_reg_tot.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0][0]
print("-----ALL ATTRIBUTE-----: Coefficient of determination: ", r_sq)
print("-----ALL ATTRIBUTE-----: slope: ", lin_reg_tot.coef_)
r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(lin_reg_tot, test_set, test_result)
print("-----ALL ATTRIBUTE-----: ADJ_R2Score: ", adj_r_squared)
print("-----ALL ATTRIBUTE-----: MSE : ",MSE)
print(("MSE: {}".format(mean_squared_error(pred, test_result))))
"""plt.scatter(test_set, test_result, color='black')
plt.plot(test_set, pred, color='blue', linewidth=3)
plt.show()

theta_0 = lin_reg_tot.intercept_
theta_1_500 = lin_reg_tot.coef_
theta_1_500 = np.array(theta_1_500)

plt.figure()
plt.plot(train_set,train_result,'x')
x = [[2933, 2928, 1, 2015, 2015, 100, 200, 69, 3, 10]]
print("x",x)
plt.plot(x,theta_0+x*theta_1_500,'r')
plt.show()"""
