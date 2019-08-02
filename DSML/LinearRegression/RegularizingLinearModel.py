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
Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/ListStudent.txt"))

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

print("--------RIDGE REGERESSION-------")
print("Coefficient of determination: ", r_sq)
print("intercept: ", rid_reg.intercept_)
print("slope: ", rid_reg.coef_)

maximum = 0.0
sw = 0
minus = 0.0
School = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictSchool.txt"))
Name_School = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictCodSchool.txt"))
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    predictiveTemp = [[tipo_maturità, voto_diploma]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = rid_reg.predict(predictiveTemp)
    y_pred = float(y_pred)
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
max_school = str(max_school)
min_school = str(min_school)

name_max_school = School.get(max_school, None)
name_min_school = School.get(min_school, None)

print("Highest score: ", maximum, sep="\n")
print("Worst score: ", minus,  sep="\n")


print("--------ELASTIC NET-------")
r_sq = elastic_net.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", elastic_net.intercept_)
print("slope: ", elastic_net.coef_)

maximum = 0.0
sw = 0
minus = 0.0
School = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictSchool.txt"))
Name_School = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictCodSchool.txt"))
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    predictiveTemp = [[tipo_maturità, voto_diploma]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = elastic_net.predict(predictiveTemp)
    y_pred = float(y_pred)
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

max_school = str(max_school)
min_school = str(min_school)
name_max_school = School.get(max_school, None)
name_min_school = School.get(min_school, None)

print("Highest score: ", maximum, sep="\n")
print("Worst score: ", minus, sep="\n")


print("--------SGD REGRESSOR-------")
r_sq = sgd_reg.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", sgd_reg.intercept_)
print("slope: ", sgd_reg.coef_)

maximum = 0.0
sw = 0
minus = 0.0
School = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictSchool.txt"))
Name_School = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictCodSchool.txt"))
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])

    predictiveTemp = [[tipo_maturità, voto_diploma]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = sgd_reg.predict(predictiveTemp)
    y_pred = float(y_pred)
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
max_school = str(max_school)
min_school = str(min_school)

name_max_school = School.get(max_school, None)
name_min_school = School.get(min_school, None)
print("Highest score: ", maximum, sep="\n")
print("Worst score: ", minus,sep="\n")

print("--------Logistic Regression-------")
r_sq = log_reg.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", log_reg.intercept_)
print("slope: ", log_reg.coef_)
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])

    predictiveTemp = [[tipo_maturità, voto_diploma]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = sgd_reg.predict(predictiveTemp)
    y_pred = float(y_pred)
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

print("Highest score: ", maximum, sep="\n")
print("Worst score: ", minus,sep="\n")