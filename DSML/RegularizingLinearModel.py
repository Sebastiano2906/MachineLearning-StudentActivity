import numpy as np
from sklearn.linear_model import LinearRegression
import json
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("ListStudent.txt"))

Train_size = int((len(Maturità) / 100) * 80)
Test_size = int((len(Maturità)/100) * 20)
Train_set = []
Test_set = []
Result_Test = []
for i in range(0, Train_size):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    nome_scuola = int(Maturità[i][0][3])
    TrainTemp = [tipo_maturità, voto_diploma, nome_scuola]
    Result.append(int(Maturità[i][0][2]))
    Train_set.append(TrainTemp)
for i in range(Train_size+1, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    nome_scuola = int(Maturità[i][0][3])
    TestTemp = [tipo_maturità, voto_diploma, nome_scuola]
    Result_Test.append(int(Maturità[i][0][2]))
    Test_set.append(TestTemp)

print(len(Train_set))
print(len(Test_set))
Train_set, Result = np.array(Train_set), np.array(Result)
rid_reg = Ridge(alpha=1, solver="cholesky")
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(Train_set, Result.ravel())
rid_reg.fit(Train_set, Result)
elastic_net.fit(Train_set,Result)
r_sq = rid_reg.score(Train_set, Result)

print("--------RIDGE REGERESSION-------")
print("Coefficient of determination: ", r_sq)
print("intercept: ", rid_reg.intercept_)
print("slope: ", rid_reg.coef_)

maximum = 0.0
sw = 0
minus = 0.0
School = json.load(open("DictSchool.txt"))
Name_School = json.load(open("DictCodSchool.txt"))
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    nome_scuola = int(item[2])
    predictiveTemp = [[tipo_maturità, voto_diploma, nome_scuola]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = rid_reg.predict(predictiveTemp)
    y_pred = float(y_pred)
    if sw == 0:
        minus = y_pred
        min_school = tipo_maturità
        min_name_school = nome_scuola
        sw = 1
    elif minus > y_pred:
        minus = y_pred
        min_name_school = nome_scuola
        min_school = tipo_maturità
    if y_pred > maximum:
        maximum = y_pred
        max_school = tipo_maturità
        max_name_school = nome_scuola
max_school = str(max_school)
min_school = str(min_school)
min_name_school = str(min_name_school)
max_name_school = str(max_name_school)
name_max_school = School.get(max_school, None)
name_min_school = School.get(min_school, None)
min_name_school = Name_School.get(min_name_school, None)
max_name_school = Name_School.get(max_name_school, None)
print("Highest score: ", maximum, "of type " + name_max_school, "of school " + max_name_school, sep="\n")
print("Worst score: ", minus, "of school " + name_min_school, "of school " + min_name_school, sep="\n")


print("--------ELASTIC NET-------")
r_sq = elastic_net.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", elastic_net.intercept_)
print("slope: ", elastic_net.coef_)

maximum = 0.0
sw = 0
minus = 0.0
School = json.load(open("DictSchool.txt"))
Name_School = json.load(open("DictCodSchool.txt"))
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    nome_scuola = int(item[2])
    predictiveTemp = [[tipo_maturità, voto_diploma, nome_scuola]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = elastic_net.predict(predictiveTemp)
    y_pred = float(y_pred)
    if sw == 0:
        minus = y_pred
        min_school = tipo_maturità
        min_name_school = nome_scuola
        sw = 1
    elif minus > y_pred:
        minus = y_pred
        min_name_school = nome_scuola
        min_school = tipo_maturità
    if y_pred > maximum:
        maximum = y_pred
        max_school = tipo_maturità
        max_name_school = nome_scuola
max_school = str(max_school)
min_school = str(min_school)
min_name_school = str(min_name_school)
max_name_school = str(max_name_school)
name_max_school = School.get(max_school, None)
name_min_school = School.get(min_school, None)
min_name_school = Name_School.get(min_name_school, None)
max_name_school = Name_School.get(max_name_school, None)
print("Highest score: ", maximum, "of type " + name_max_school, "of school " + max_name_school, sep="\n")
print("Worst score: ", minus, "of school " + name_min_school, "of school " + min_name_school, sep="\n")


print("--------SGD REGRESSOR-------")
r_sq = sgd_reg.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", sgd_reg.intercept_)
print("slope: ", sgd_reg.coef_)

maximum = 0.0
sw = 0
minus = 0.0
School = json.load(open("DictSchool.txt"))
Name_School = json.load(open("DictCodSchool.txt"))
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    nome_scuola = int(item[2])
    predictiveTemp = [[tipo_maturità, voto_diploma, nome_scuola]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = sgd_reg.predict(predictiveTemp)
    y_pred = float(y_pred)
    if sw == 0:
        minus = y_pred
        min_school = tipo_maturità
        min_name_school = nome_scuola
        sw = 1
    elif minus > y_pred:
        minus = y_pred
        min_name_school = nome_scuola
        min_school = tipo_maturità
    if y_pred > maximum:
        maximum = y_pred
        max_school = tipo_maturità
        max_name_school = nome_scuola
max_school = str(max_school)
min_school = str(min_school)
min_name_school = str(min_name_school)
max_name_school = str(max_name_school)
name_max_school = School.get(max_school, None)
name_min_school = School.get(min_school, None)
min_name_school = Name_School.get(min_name_school, None)
max_name_school = Name_School.get(max_name_school, None)
print("Highest score: ", maximum, "of type " + name_max_school, sep="\n")
print("Worst score: ", minus, "of school " + name_min_school, sep="\n")


