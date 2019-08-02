"""
Implementazione di una RandomForest.
Sono ancora in fase di seminazione degli alberi. Lasciate che la natura faccia il suo corso, e quando
cresceranno gli alberelli vi spiegherò.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import json

Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("ListStudentAug.txt"))
Train_size = int((len(Maturità) / 100) * 80)
Test_size = int((len(Maturità)/100) * 20)
Train_set = []
Test_set = []
Result_Test = []
Total_Set = []
Tota_Result = []

for i in range(1, Train_size):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TrainTemp = [tipo_maturità, voto_diploma]
    Result.append(int(Maturità[i][0][2]))
    Train_set.append(TrainTemp)

for i in range(Train_size+1, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TrainTemp = [tipo_maturità, voto_diploma]
    Result_Test.append(int(Maturità[i][0][2]))
    Test_set.append(TrainTemp)

from sklearn.ensemble import  RandomForestRegressor

rnd_reg = RandomForestRegressor(n_estimators=200, max_leaf_nodes=8, n_jobs=-1)
rnd_reg.fit(Train_set, Result)
from sklearn.metrics import mean_squared_error
print("Score: ", rnd_reg.score(Test_set, Result_Test))
y_pred_pred = []
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    predictiveTemp = [[tipo_maturità, voto_diploma]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = rnd_reg.predict(predictiveTemp)
    y_pred = float(y_pred)
    y_pred_pred.append(y_pred)
print("MSE: ", mean_squared_error(y_pred_pred,Result_Test))