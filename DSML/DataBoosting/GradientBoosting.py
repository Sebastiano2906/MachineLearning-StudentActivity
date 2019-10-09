import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

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
reg = GradientBoostingRegressor(n_estimators=3, learning_rate=1)

reg.fit(train_set_tot, train_result_tot.ravel())

print("Score: ", reg.score(test_set_tot, test_result_tot.ravel()))
print("Feature importance: ", reg.feature_importances_)
newStudent = [[1999, 2928, 1, 2018, 2018, 60, 54, 9, 1, 2]]
real_value = [30]
print("Predicted: ", reg.predict(newStudent))