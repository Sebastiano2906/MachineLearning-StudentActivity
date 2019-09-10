"""
Implementazione di una RandomForest.
Sono ancora in fase di seminazione degli alberi. Lasciate che la natura faccia il suo corso, e quando
cresceranno gli alberelli vi spiegherò.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set = []
test_set = []
train_result = []
test_result = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][12]])
        train_result.append([predictiveAttributeDegree[i][2]])
    else:
        test_set.append([predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][12]])
        test_result.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][12]])
        train_result.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][12]])
        test_result.append([predictiveAttributeNotDegree[i][2]])

train_result = np.array(train_result)
rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=10, n_jobs=-1, max_depth=10)
#print(cross_val_score(rnd_reg, train_set[1:], train_result[1:], cv=10))
rnd_reg.fit(train_set, train_result.ravel())
print(rnd_reg.score(test_set, test_result))
#              matr cf    2  3 tot cds tipoCds coorte annicarriera annodiploma votodip codschool tipoMat annolaur votolaur erasmus tesi mot_sta sta fc
newStudent = [[100, 11]]
real_value = [40]
predicted = rnd_reg.predict(newStudent)
print("Predicted: ", predicted)
print("MSE: ", mean_squared_error(real_value, rnd_reg.predict(newStudent)))
print("Params: ", rnd_reg.get_params())
print("Feature Importance: ", rnd_reg.feature_importances_)

print("\n\n\n----------QUI INIZIA LA SEZIONE CON TUTTI GLI ATTRIBUTI---------- \n\n\n")
predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set_tot = []
test_set_tot = []
train_result_tot = []
test_result_tot = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set_tot.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10],
                          predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][17],
                          predictiveAttributeNotDegree[i][18]])
        train_result_tot.append([predictiveAttributeDegree[i][2]])
    else:
        test_set_tot.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10],
                          predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][17],
                          predictiveAttributeNotDegree[i][18]])
        test_result_tot.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set_tot.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10],
                          predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][17],
                          predictiveAttributeNotDegree[i][18]])
        train_result_tot.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set_tot.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10],
                          predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][17],
                          predictiveAttributeNotDegree[i][18]])
        test_result_tot.append([predictiveAttributeNotDegree[i][2]])
train_result_tot = np.array(train_result_tot)
rnd_reg_AllAttribute = RandomForestRegressor(n_estimators=500, max_leaf_nodes=10, n_jobs=-1, max_depth=10)
rnd_reg_AllAttribute.fit(train_set_tot, train_result_tot.ravel())

print(rnd_reg_AllAttribute.score(test_set_tot, test_result_tot))
#              0. matr 1.cf  6.tipoCds  7.coorte  9.annodiploma 10.votodip 11.codschool 12.tipoMat  17.mot_sta 18.sta
newStudent = [[1999, 2928, 1, 2018, 2018, 60, 54, 9, 1, 2]]
real_value = [30]
predicted = rnd_reg_AllAttribute.predict(newStudent)
print("---ALL ATTRIBUTE----: Predicted: ", predicted)
print("---ALL ATTRIBUTE----: MSE: ", mean_squared_error(real_value, rnd_reg_AllAttribute.predict(newStudent)))
print("---ALL ATTRIBUTE----: Params: ", rnd_reg_AllAttribute.get_params())
print("---ALL ATTRIBUTE----: Feature Importance: ", rnd_reg_AllAttribute.feature_importances_)
