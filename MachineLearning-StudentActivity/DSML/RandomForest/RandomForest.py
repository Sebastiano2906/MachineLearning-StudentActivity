

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


predictiveAttributeDegree = pd.read_json("../MachineLearning-StudentActivity/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("../MachineLearning-StudentActivity/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set = []
test_set = []
train_result = []
test_result = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][13]])
        train_result.append([predictiveAttributeDegree[i][2]])
    else:
        test_set.append([predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][13]])
        test_result.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][13]])
        train_result.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][13]])
        test_result.append([predictiveAttributeNotDegree[i][2]])

train_result = np.array(train_result)
rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=10, n_jobs=-1, max_depth=10)
#print(cross_val_score(rnd_reg, train_set[1:], train_result[1:], cv=10))
rnd_reg.fit(train_set, train_result.ravel())
print(rnd_reg.score(test_set, test_result))
prediction = []
for item in test_set:
    items = [[item[0], item[1]]]
    prediction.append(rnd_reg.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
print(("MSE: {}".format(mean_squared_error(pred, test_result))))
print("Params: ", rnd_reg.get_params())
print("Feature Importance: ", rnd_reg.feature_importances_)



print("\n\n\n----------QUI INIZIA LA SEZIONE CON TUTTI GLI ATTRIBUTI---------- \n\n\n")
predictiveAttributeDegree = pd.read_json("../MachineLearning-StudentActivity/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("../MachineLearning-StudentActivity/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set_tot = []
test_set_tot = []
train_result_tot = []
test_result_tot = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set_tot.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        train_result_tot.append([predictiveAttributeDegree[i][2]])
    else:
        test_set_tot.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        test_result_tot.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set_tot.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8], predictiveAttributeNotDegree[i][9],
                          predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][17]])
        train_result_tot.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set_tot.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8], predictiveAttributeNotDegree[i][9],
                          predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][17]])
        test_result_tot.append([predictiveAttributeNotDegree[i][2]])
train_result_tot = np.array(train_result_tot)
test_result_tot = np.array(test_result_tot)
rnd_reg_AllAttribute = RandomForestRegressor(n_estimators=500, max_leaf_nodes=10, n_jobs=-1, max_depth=10)
rnd_reg_AllAttribute.fit(train_set_tot, train_result_tot.ravel())

print(rnd_reg_AllAttribute.score(test_set_tot, test_result_tot.ravel()))
prediction = []
for item in test_set_tot:
    items = [[item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10]]]
    prediction.append(rnd_reg_AllAttribute.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
print(("MSE: {}".format(mean_squared_error(pred, test_result_tot))))
print("---ALL ATTRIBUTE----: Params: ", rnd_reg_AllAttribute.get_params())
print("---ALL ATTRIBUTE----: Feature Importance: ", rnd_reg_AllAttribute.feature_importances_)
