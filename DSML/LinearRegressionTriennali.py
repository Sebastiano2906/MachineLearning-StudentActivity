import numpy as np
from sklearn.linear_model import LinearRegression
import json

Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("ListStudent.txt"))
for i in range(0, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    predictiveTemp = [tipo_maturità, voto_diploma]
    Result.append(int(Maturità[i][0][2]))
    Predictive_Value.append(predictiveTemp)

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
