import numpy as np
from sklearn.linear_model import LinearRegression
import json

Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("ListStudent.txt"))
"""
for i in range(0, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    predictiveTemp = [tipo_maturità, voto_diploma]
    Result.append(int(Maturità[i][0][2]))
    Predictive_Value.append(predictiveTemp)
"""
Train_size = int((len(Maturità) / 100) * 70)
Test_size = int((len(Maturità)/100) * 30)
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

Train_set, Result = np.array(Train_set), np.array(Result)
model = LinearRegression().fit(Train_set, Result)
r_sq = model.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", model.intercept_)
print("slope: ", model.coef_)
"""
count = 0
for i in range(0, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    predictiveTemp = [[tipo_maturità, voto_diploma]]
    y_pred = int(model.predict(predictiveTemp))
    CFU = int(Maturità[i][0][2])
    diff = CFU - y_pred
    if diff < 9 :
        count +=1
"""
#accuracy = (count/len(Maturità)) * 100
#print("Precisione", accuracy, "%")
max = 0.0
School = json.load(open("DictSchool.txt"))
print(School)
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    predictiveTemp = [[tipo_maturità, voto_diploma]]
    y_pred = model.predict(predictiveTemp)
    y_pred = float(y_pred)
    if y_pred > max:
        max = y_pred
        max_school = tipo_maturità
max_school = str(max_school)
name_max_school = School.get(max_school, None)
print("Highest score: ", max, "of school " + name_max_school, sep="\n")

"""  
Predictive_Value, Result = np.array(Predictive_Value), np.array(Result)
model = LinearRegression().fit(Predictive_Value, Result)
r_sq = model.score(Predictive_Value, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", model.intercept_)
print("slope: ", model.coef_)
count = 0
for i in range(0, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    predictiveTemp = [[tipo_maturità, voto_diploma]]
    y_pred = int(model.predict(predictiveTemp))
    CFU = int(Maturità[i][0][2])
    diff = CFU - y_pred
    if diff < 9 :
        count +=1

accuracy = (count/len(Maturità)) * 100
print("Precisione", accuracy, "%")
Test_Predict = [[56341636, 100]]
y_pred = model.predict(Test_Predict)
print("Predicted response:", y_pred, sep="\n")
"""