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

