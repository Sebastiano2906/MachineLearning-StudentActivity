from sklearn.svm import LinearSVR
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
predictiveAttributeDegree = pd.read_json("../predictiveDegree.txt", orient='records', dtype=True,typ="series") #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
predictiveAttributeNotDegree = pd.read_json("../predictiveNotDegree.txt", orient='records', dtype=True,typ="series") #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE


train_set_tot = []
test_set_tot = []
train_result_tot = []
test_result_tot = []
count = 0
svm_reg_tot = LinearSVR(epsilon=1.0, max_iter=100000000)
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
svm_reg_tot.fit(train_set_tot, train_result_tot.ravel())

print("----ALL ATTRIBUTE: score: ", svm_reg_tot.score(test_set_tot, test_result_tot))
#              0. matr 1.cf  6.tipoCds  7.coorte  9.annodiploma 10.votodip 11.codschool 12.tipoMat  17.mot_sta 18.sta
newStudent = [[2933, 2928, 1, 2015, 2015, 100, 200, 9, 3, 10]]
real_value = [30]
predicted = svm_reg_tot.predict(newStudent)

print("----ALL ATTRIBUTE: Predicted: ", predicted)
print("----ALL ATTRIBUTE: MSE: ", mean_squared_error(real_value, svm_reg_tot.predict(newStudent)))
print("----ALL ATTRIBUTE: Params: ", svm_reg_tot.get_params())