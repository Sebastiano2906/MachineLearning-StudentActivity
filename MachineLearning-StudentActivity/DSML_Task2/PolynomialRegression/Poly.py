from sklearn.preprocessing import PolynomialFeatures
import json
import numpy as np
import csv
from sklearn.linear_model import LinearRegression

#Maturità = json.load(open("../ListaStudentiModificata.txt"))
Maturità = [tuple(row) for row in csv.reader(open("../DatasetTriennaliNuovo.csv", 'r'))] #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
Train_size = int((len(Maturità) / 100) * 80)
Test_size = int((len(Maturità)/100) * 20)
Train_set = []
Test_set = []
Result_Test = []
Result = []
for i in range(1, Train_size):
    tipo_maturità = int(Maturità[i][21])  # primo volore predittivo
    voto_diploma = int(Maturità[i][11])  # secondo valore predittivo
    cfu_primo = int(Maturità[i][2])  # terzo valore predittivo
    #cfu_secodno = int(Maturità[i][0][3])
    TrainTemp = [tipo_maturità, voto_diploma, cfu_primo]
    Result.append(int(Maturità[i][20]))
    Train_set.append(TrainTemp)
for i in range(Train_size+1, len(Maturità)):
    tipo_maturità = int(Maturità[i][21])  # primo volore predittivo
    voto_diploma = int(Maturità[i][11])  # secondo valore predittivo
    cfu_primo = int(Maturità[i][2])  # terzo valore predittivo

    TestTemp = [tipo_maturità, voto_diploma, cfu_primo]
    Result_Test.append(int(Maturità[i][20]))
    Test_set.append(TestTemp)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
Train_set, Result, Test_set, Result_Test = np.array(Train_set), np.array(Result), np.array(Test_set), np.array(Result_Test)
print(Test_set)
Train_set_poly = poly_features.fit_transform(Train_set)
model = LinearRegression().fit(Train_set_poly, Result)
r_sq = model.score(Train_set_poly, Result)
print("Coefficient of determination: ", r_sq)
print("Intercept: ", model.intercept_)
print("Slope:", model.coef_)






maximum = 0.0
sw = 0
minus = 0.0
y_pred_pred = []
School = json.load(open("../Tipo_mat.txt")) #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
for item in Test_set:
    tipo_maturità = item[0]
    #print("-------------------------------",tipo_maturità)
    voto_diploma = int(item[1])
    cfu_primo= int(item[2])
   # cfu_secodno=int(item[3])
    predictiveTemp = [[tipo_maturità, voto_diploma, cfu_primo]]
    predictiveTemp = np.array(predictiveTemp)
    predictiveTemp = poly_features.fit_transform(predictiveTemp)
    y_pred = model.predict(predictiveTemp)
    y_pred = float(y_pred)
    y_pred_pred.append(y_pred)
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
print("Highest score: ", maximum,"school", tipo_maturità, sep="\n")
print("Worst score: ", minus,"school", tipo_maturità,  sep="\n")
from sklearn.metrics import r2_score


print("R2Score : ", r2_score(Result_Test, y_pred_pred))

"""newStudent = [3, 90, 20, 0]
real_value = [2]
predicted = model.predict(newStudent)
print("Predicted: ", predicted)"""