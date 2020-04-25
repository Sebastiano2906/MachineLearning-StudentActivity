# """
# Implementazione di una RandomForest.
# Sono ancora in fase di seminazione degli alberi. Lasciate che la natura faccia il suo corso, e quando
# cresceranno gli alberelli vi spiegherò.
# """
#
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import cross_val_score
#
#
# predictiveAttributeDegree = pd.read_json("C:/Users/salva/Desktop/Università/Magistrale - Data Science/Data Science and Machine Learning/MachineLearning-StudentActivity/DSML_Task3/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
# predictiveAttributeNotDegree = pd.read_json("C:/Users/salva/Desktop/Università/Magistrale - Data Science/Data Science and Machine Learning/MachineLearning-StudentActivity/DSML_Task3/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")
#
#
# train_set = []
# test_set = []
# train_result = []
# test_result = []
# count = 0
# train_percent = (len(predictiveAttributeDegree)/100)*80
# for i in range(len(predictiveAttributeDegree)):
#     if count < train_percent:
#         count = count + 1
#         train_set.append([predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][12], predictiveAttributeDegree[i][2]])
#         train_result.append([predictiveAttributeDegree[i][20]])
#     else:
#         test_set.append([predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][12], predictiveAttributeDegree[i][2]])
#         test_result.append([predictiveAttributeDegree[i][20]])
# train_percent = (len(predictiveAttributeNotDegree)/100)*80
# count = 0
# for i in range(len(predictiveAttributeNotDegree)):
#     if count < train_percent:
#         count = count + 1
#         train_set.append([predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][12], predictiveAttributeNotDegree[i][2]])
#         train_result.append([predictiveAttributeNotDegree[i][20]])
#     else:
#         test_set.append([predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][12],predictiveAttributeNotDegree[i][2]])
#         test_result.append([predictiveAttributeNotDegree[i][20]])
#
# train_result = np.array(train_result)
# rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=10, n_jobs=-1, max_depth=10)
# #print(cross_val_score(rnd_reg, train_set[1:], train_result[1:], cv=10))
# rnd_reg.fit(train_set, train_result.ravel())
# print(rnd_reg.score(test_set, test_result))
# #              matr cf    2  3 tot cds tipoCds coorte annicarriera annodiploma votodip codschool tipoMat annolaur votolaur erasmus tesi mot_sta sta fc
# newStudent = [[70, 3, 20]]
# real_value = [2]
# predicted = rnd_reg.predict(newStudent)
# print("Predicted: ", predicted)
# print("MSE: ", mean_squared_error(real_value, rnd_reg.predict(newStudent)))
# print("Params: ", rnd_reg.get_params())
# print("Feature Importance: ", rnd_reg.feature_importances_)
#
#

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


predictiveAttributeDegree = pd.read_json("../DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series") #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
predictiveAttributeNotDegree = pd.read_json("../DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series") #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE


train_set = []
test_set = []
train_result = []
test_result = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][2]])
        train_result.append([predictiveAttributeDegree[i][20]])
    else:
        test_set.append([predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][2]])
        test_result.append([predictiveAttributeDegree[i][20]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][2]])
        train_result.append([predictiveAttributeNotDegree[i][20]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][2]])
        test_result.append([predictiveAttributeNotDegree[i][20]])

train_result = np.array(train_result)
rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=10, n_jobs=-1, max_depth=10)
#print(cross_val_score(rnd_reg, train_set[1:], train_result[1:], cv=10))
rnd_reg.fit(train_set, train_result.ravel())
print(rnd_reg.score(test_set, test_result))
#              matr cf    2  3 tot cds tipoCds coorte annicarriera annodiploma votodip codschool tipoMat annolaur votolaur erasmus tesi mot_sta sta fc
newStudent = [[90, 60]]
real_value = [1]
predicted = rnd_reg.predict(newStudent)
print("Predicted: ", predicted)
print("MSE: ", mean_squared_error(real_value, rnd_reg.predict(newStudent)))
print("Params: ", rnd_reg.get_params())
print("Feature Importance: ", rnd_reg.feature_importances_)

