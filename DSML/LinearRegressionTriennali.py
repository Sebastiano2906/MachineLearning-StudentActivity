import numpy as np
from sklearn.linear_model import LinearRegression
import json
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("ListStudent.txt"))
Train_size = int((len(Maturità) / 100) * 80)
Test_size = int((len(Maturità)/100) * 20)
Train_set = []
Test_set = []
Result_Test = []
Total_Set = []
Tota_Result = []
for i in range(0, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    nome_scuola = int(Maturità[i][0][3])
    TrainTemp = [tipo_maturità, voto_diploma, nome_scuola]
    Tota_Result.append(int(Maturità[i][0][2]))
    Result.append(int(Maturità[i][0][2]))
    Total_Set.append(TrainTemp)
    Train_set.append(TrainTemp)


Train_set, Test_set, Result,  Result_Test= train_test_split(Train_set, Result, test_size=0.3)
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
School = json.load(open("DictSchool.txt"))
print(len(School))
Name_School = json.load(open("DictCodSchool.txt"))
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    nome_scuola = int(item[2])
    predictiveTemp = [[tipo_maturità, voto_diploma, nome_scuola]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = model.predict(predictiveTemp)
    y_pred = float(y_pred)
    if y_pred > 0:
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
