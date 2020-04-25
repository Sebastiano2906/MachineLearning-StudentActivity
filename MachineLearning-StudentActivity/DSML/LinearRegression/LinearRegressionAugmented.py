"""
Linear Regression lanciata sul dataset aumentato. Risultati pessimi.
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
Maturità = json.load(open("../MachineLearning-StudentActivity/DSML/FileGenerated/ListDatasetAug.txt"))
Train_size = int((len(Maturità) / 100) * 70)
Test_size = int((len(Maturità)/100) * 30)
Train_set = []
Test_set = []
Result_Test = []
Total_Set = []
Tota_Result = []

for i in range(0, Train_size):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TrainTemp = [tipo_maturità, voto_diploma]
    Total_Temp = [tipo_maturità, voto_diploma, int(Maturità[i][0][2])]
    Tota_Result.append(int(Maturità[i][0][2]))
    Result.append(int(Maturità[i][0][2]))
    Total_Set.append(TrainTemp)
    Train_set.append(TrainTemp)


for i in range(Train_size+1, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TrainTemp = [tipo_maturità, voto_diploma]
    Total_Temp = [tipo_maturità, voto_diploma, int(Maturità[i][0][2])]
    Tota_Result.append(int(Maturità[i][0][2]))
    Result_Test.append(int(Maturità[i][0][2]))
    Total_Set.append(TrainTemp)
    Test_set.append(TrainTemp)



#Train_set, Test_set, Result,  Result_Test= train_test_split(Train_set, Result, test_size=0.3)
Train_set, Result = np.array(Train_set), np.array(Result)
Test_set, Result_Test = np.array(Test_set), np.array(Result_Test)
model = LinearRegression()
model.fit(Train_set, Result)
r_sq = model.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", model.intercept_)
print("slope: ", model.coef_)
maximum = 0.0
sw = 0
minus = 0.0
School = json.load(open("../MachineLearning-StudentActivity/DSML/FileGenerated/DictSchool.txt"))
Name_School = json.load(open("../MachineLearning-StudentActivity/DSML/FileGenerated/DictCodSchool.txt"))
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
max_school = str(max_school)
min_school = str(min_school)
name_max_school = School.get(max_school, None)
name_min_school = School.get(min_school, None)
print("Highest score: ", maximum, "of type ", sep="\n")
print("Worst score: ", minus, "of school ", sep="\n")
from sklearn.metrics import r2_score
from sklearn.metrics import  mean_squared_error
print("MEAN_SQUARED_ERROR: ", mean_squared_error(Result_Test, y_pred_pred))
print("R2Score : ", r2_score(Result_Test, y_pred_pred))

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = np.array(Train_set), np.array(Test_set), np.array(Result), np.array(Result_Test)

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
