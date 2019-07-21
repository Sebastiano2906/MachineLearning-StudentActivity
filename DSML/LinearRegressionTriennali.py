import numpy as np
from sklearn.linear_model import LinearRegression
import json
from matplotlib import pyplot as plt

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
Train_size = int((len(Maturità) / 100) * 75)
Test_size = int((len(Maturità)/100) * 25)
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
    y_pred = model.predict(predictiveTemp)
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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
plot_learning_curves(model, Train_set, Result)
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