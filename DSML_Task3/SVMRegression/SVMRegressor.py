from sklearn.svm import LinearSVR
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
predictiveAttributeDegree = pd.read_json("C:/Users/salva/Desktop/Università/Magistrale - Data Science/Data Science and Machine Learning/MachineLearning-StudentActivity/DSML_Task3/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/salva/Desktop/Università/Magistrale - Data Science/Data Science and Machine Learning/MachineLearning-StudentActivity/DSML_Task3/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set = []
test_set = []
train_result = []
test_result = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][12], predictiveAttributeDegree[i][2]])
        train_result.append([predictiveAttributeDegree[i][20]])
    else:
        test_set.append([predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][12], predictiveAttributeDegree[i][2]])
        test_result.append([predictiveAttributeDegree[i][20]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][12], predictiveAttributeNotDegree[i][2]])
        train_result.append([predictiveAttributeNotDegree[i][20]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][12], predictiveAttributeNotDegree[i][2]])
        test_result.append([predictiveAttributeNotDegree[i][20]])

svm_reg = LinearSVR(epsilon=1.0, max_iter=10000000)
train_result = np.array(train_result)
svm_reg.fit(train_set, train_result.ravel())
print(svm_reg.score(test_set, test_result))
newStudent = [[100,9,40]]
real_value = [0]
predicted = svm_reg.predict(newStudent)

print("Predicted: ", predicted)
print("MSE: ", mean_squared_error(real_value, svm_reg.predict(newStudent)))
print("Params: ", svm_reg.get_params())









