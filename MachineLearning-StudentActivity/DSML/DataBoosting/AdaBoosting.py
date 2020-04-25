import pandas as pd
import numpy as np
from sklearn.ensemble import  AdaBoostRegressor

predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set_tot = []
test_set_tot = []
train_result_tot = []
test_result_tot = []
count = 0
cfu = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set_tot.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        cfu = int(predictiveAttributeDegree[i][2])
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
reg = AdaBoostRegressor(n_estimators=4, random_state=0)
reg.fit(train_set_tot, train_result_tot.ravel())

print("Score: ", reg.score(test_set_tot, test_result_tot.ravel()))
print("Feature importance: ", reg.feature_importances_)
newStudent = [[1999, 2928, 1, 2018, 2018, 60, 54, 9, 1, 2]]
real_value = [30]
print("Predicted: ", reg.predict(newStudent))


print("--------------ALTRA TECNICA--------------")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve


score = []
"""for i in range(0, len(train_result_tot)):
    if train_result_tot[i] <= 15:
        train_result_tot[i] = 0
    elif train_result_tot[i] <= 30:
        train_result_tot[i] = 1
    elif train_result_tot[i] <= 45:
        train_result_tot[i] = 2
    else:
        train_result_tot[i] = 3
"""
for i in range(0, len(train_result_tot)):
    if train_result_tot[i] <= 30:
        train_result_tot[i] = 0
    else:
        train_result_tot[i] = 1
#"""

reg_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
scores_ada = cross_val_score(reg_ada, train_set_tot, train_result_tot.ravel(), cv=2)
scores_ada.mean()
print("Scores : ", scores_ada)

for depth in [1,2,10] :
    reg_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))
    scores_ada = cross_val_score(reg_ada, train_set_tot, train_result_tot.ravel(), cv=10)
    score.append(scores_ada.mean())

newStudent = [[1999, 2928, 1, 2018, 2018, 60, 54, 9, 1, 2]]
real_value = [30]
reg_ada.fit(train_set_tot,train_result_tot)
print("Predicted: ", cross_val_predict(reg_ada, test_set_tot, test_result_tot, cv=2))
print("Score: ", score)