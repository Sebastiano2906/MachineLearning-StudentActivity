from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
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
        train_set.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][3], predictiveAttributeDegree[i][4],
                          predictiveAttributeDegree[i][5], predictiveAttributeDegree[i][6], predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8],
                          predictiveAttributeDegree[i][9], predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][14], predictiveAttributeDegree[i][15], predictiveAttributeDegree[i][16],
                          predictiveAttributeDegree[i][17], predictiveAttributeDegree[i][18], predictiveAttributeDegree[i][19], predictiveAttributeDegree[i][20]])
        train_result.append([predictiveAttributeDegree[i][2]])
    else:
        test_set.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][3], predictiveAttributeDegree[i][4],
                          predictiveAttributeDegree[i][5], predictiveAttributeDegree[i][6], predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8],
                          predictiveAttributeDegree[i][9], predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][14], predictiveAttributeDegree[i][15], predictiveAttributeDegree[i][16],
                          predictiveAttributeDegree[i][17], predictiveAttributeDegree[i][18], predictiveAttributeDegree[i][19], predictiveAttributeDegree[i][20]])
        test_result.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][3], predictiveAttributeNotDegree[i][4],
                          predictiveAttributeNotDegree[i][5], predictiveAttributeNotDegree[i][6], predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8],
                          predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][14], predictiveAttributeNotDegree[i][15], predictiveAttributeNotDegree[i][16],
                          predictiveAttributeNotDegree[i][17], predictiveAttributeNotDegree[i][18], predictiveAttributeNotDegree[i][19], predictiveAttributeNotDegree[i][20]])
        train_result.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][3], predictiveAttributeNotDegree[i][4],
                          predictiveAttributeNotDegree[i][5], predictiveAttributeNotDegree[i][6], predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8],
                          predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][14], predictiveAttributeNotDegree[i][15], predictiveAttributeNotDegree[i][16],
                          predictiveAttributeNotDegree[i][17], predictiveAttributeNotDegree[i][18], predictiveAttributeNotDegree[i][19], predictiveAttributeNotDegree[i][20]])
        test_result.append([predictiveAttributeNotDegree[i][2]])

svm_reg = SVR(kernel="poly", degree=2, C=50, epsilon=1.0, gamma="auto")
train_result = np.array(train_result)
svm_reg.fit(train_set, train_result.ravel())

print(svm_reg.score(test_set, test_result))
newStudent = [[2933, 2928, 0, 0, 30, 1, 1, 2018, 1, 2018, 100, 200, 1, 0, 0, 0, 0, 1, 2, 0]]
real_value = [30]
predicted = svm_reg.predict(newStudent)

print("Predicted: ", predicted)
print("MSE: ", mean_squared_error(real_value, svm_reg.predict(newStudent)))
print("Params: ", svm_reg.get_params())

