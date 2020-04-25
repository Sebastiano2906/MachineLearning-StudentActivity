"""
Qui vengono implementati alcuni modelli presi dal libro per cercare di innalzare il punteggio R2Score della linearRegression
Tuttavia risultati comunque pessimi.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import json
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("../MachineLearning-StudentActivity/DSML/FileGenerated/ListStudent.txt"))

Train_size = int((len(Maturità) / 100) * 80)
Test_size = int((len(Maturità)/100) * 20)
Train_set = []
Test_set = []
Result_Test = []
for i in range(0, Train_size):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TrainTemp = [tipo_maturità, voto_diploma]
    Result.append(int(Maturità[i][0][2]))
    Train_set.append(TrainTemp)
for i in range(Train_size+1, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TestTemp = [tipo_maturità, voto_diploma]
    Result_Test.append(int(Maturità[i][0][2]))
    Test_set.append(TestTemp)

print(len(Train_set))
print(len(Test_set))
Train_set, Result = np.array(Train_set), np.array(Result)
rid_reg = Ridge(alpha=1, solver="cholesky")
log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(Train_set, Result)
rid_reg.fit(Train_set, Result)
elastic_net.fit(Train_set, Result)
log_reg.fit(Train_set, Result)
r_sq = rid_reg.score(Train_set, Result)
newStudent = [[9,92]]
realValue = [42]

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




print("--------ELASTIC NET-------")
r_sq = elastic_net.score(Test_set, Result_Test)
print("Coefficient of determination: ", r_sq)
print("intercept: ", elastic_net.intercept_)
print("slope: ", elastic_net.coef_)
prediction = []
for item in Test_set:
    items = [[item[0], item[1]]]
    prediction.append(elastic_net.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(elastic_net, Test_set, Result_Test)
print("ADJ_R2Score: ", adj_r_squared)
print(("MSE: {}".format(mean_squared_error(pred, Result_Test))))

print("--------RIDGE REGERESSION-------")
r_sq = rid_reg.score(Test_set, Result_Test)
print("Coefficient of determination: ", r_sq)
print("intercept: ", rid_reg.intercept_)
print("slope: ", rid_reg.coef_)
prediction = []
for item in Test_set:
    items = [[item[0], item[1]]]
    prediction.append(rid_reg.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(rid_reg, Test_set, Result_Test)
print("ADJ_R2Score: ", adj_r_squared)
print(("MSE: {}".format(mean_squared_error(pred, Result_Test))))


print("--------SGD REGRESSOR-------")
r_sq = sgd_reg.score(Test_set,Result_Test)
print("Coefficient of determination: ", r_sq)
print("intercept: ", sgd_reg.intercept_)
print("slope: ", sgd_reg.coef_)
prediction = []
for item in Test_set:
    items = [[item[0], item[1]]]
    prediction.append(sgd_reg.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(sgd_reg, Test_set, Result_Test)
print("ADJ_R2Score: ", adj_r_squared)
print(("MSE: {}".format(mean_squared_error(pred, Result_Test))))

print("--------Logistic Regression-------")
r_sq = log_reg.score(Test_set, Result_Test)
print("Coefficient of determination: ", r_sq)
print("intercept: ", log_reg.intercept_)
print("slope: ", log_reg.coef_)
prediction = []
for item in Test_set:
    items = [[item[0], item[1]]]
    prediction.append(log_reg.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(log_reg, Test_set, Result_Test)
print("ADJ_R2Score: ", adj_r_squared)
print(("MSE: {}".format(mean_squared_error(pred, Result_Test))))


#   QUI COMINCIA LA PROVA CON TUTTI GLI ATTRIBUTI


predictiveAttributeDegree = pd.read_json("../MachineLearning-StudentActivity/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("../MachineLearning-StudentActivity/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


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

rid_reg = Ridge(alpha=1, solver="cholesky")
log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(train_set, train_result)
rid_reg.fit(train_set, train_result)
elastic_net.fit(train_set, train_result)
log_reg.fit(train_set, train_result)

newStudent = [[633, 1355, 1, 1, 2013, 3, 2013, 92, 54, 9, 0]]
realValue = [42]
print("-----------ALL ATTRIBUTE-----------")
print("--------ELASTIC NET-------")
r_sq = elastic_net.score(test_set, test_result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", elastic_net.intercept_)
print("slope: ", elastic_net.coef_)
prediction = []
for item in test_set:
    items = [[item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10]]]
    prediction.append(elastic_net.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
print(("MSE: {}".format(mean_squared_error(pred, test_result))))
# r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(elastic_net, test_set, test_result)
# print("ADJ_R2Score: ", adj_r_squared)
#print("MSE : ", MSE)

print("--------RIDGE REGERESSION-------")
r_sq = rid_reg.score(test_set, test_result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", rid_reg.intercept_)
print("slope: ", rid_reg.coef_)
prediction = []
for item in test_set:
    items = [[item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10]]]
    prediction.append(elastic_net.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
print(("MSE: {}".format(mean_squared_error(pred, test_result))))
# r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(rid_reg, test_set, test_result)
# print("ADJ_R2Score: ", adj_r_squared)


print("--------SGD REGRESSOR-------")
r_sq = sgd_reg.score(test_set, test_result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", sgd_reg.intercept_)
print("slope: ", sgd_reg.coef_)
prediction = []
for item in test_set:
    items = [[item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10]]]
    prediction.append(elastic_net.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
print(("MSE: {}".format(mean_squared_error(pred, test_result))))
# r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(sgd_reg, test_set, test_result)
# print("ADJ_R2Score: ", adj_r_squared)
# #print("MSE : ", MSE)

print("--------Logistic Regression-------")
r_sq = log_reg.score(test_set, test_result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", log_reg.intercept_)
print("slope: ", log_reg.coef_)
prediction = []
for item in test_set:
    items = [[item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10]]]
    prediction.append(elastic_net.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
print(("MSE: {}".format(mean_squared_error(pred, test_result))))
# r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(log_reg, test_set, test_result)
# print("ADJ_R2Score: ", adj_r_squared)
# #print("MSE : ", MSE)