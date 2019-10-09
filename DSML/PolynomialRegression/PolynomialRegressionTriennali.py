"""
Stesso discorso di LinearRegression.py. Qui viene implementato PolynomialRegression per cercare di alzare il punteggio
R2Score. Dal libro potrete leggere una spiegazione migliore di quella che io potrei mai farvi, casomai servisse.
Il grado del polinomi è fissato a 2, e non ha senso aumentarlo. Provare per credere.

Il punteggio R2Score si innalza, ma resta comunque al di sotto di 0.20 e quindi altamente inaccetabile. Motivi? Vedi
LinearRegression.py
"""

from sklearn.preprocessing import PolynomialFeatures
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error

Maturità = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/ListStudentAug.txt"))
Train_size = int((len(Maturità) / 100) * 80)
Test_size = int((len(Maturità)/100) * 20)
Train_set = []
Test_set = []
Result_Test = []
Result = []
for i in range(0, Train_size):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TrainTemp = [tipo_maturità, voto_diploma]
    Result.append(int(Maturità[i][0][2]))
    Train_set.append(TrainTemp)
for i in range(Train_size+1, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])
    voto_diploma = int(Maturità[i][0][1])
    TestTemp = [tipo_maturità, voto_diploma]
    Result_Test.append(int(Maturità[i][0][2]))
    Test_set.append(TestTemp)

newStudent = [[9,92]]
realValue = [42]
poly_features = PolynomialFeatures(degree=2, include_bias=False)
Train_set, Result, Test_set, Result_Test = np.array(Train_set), np.array(Result), np.array(Test_set), np.array(Result_Test)
Result_set_poly = Result.reshape(-1,1)
Train_set_poly = poly_features.fit_transform(Train_set)
# Result_set_poly = poly_features.fit_transform(Result_Test)
Test_set_poly = poly_features.fit_transform(Test_set)
model = LinearRegression()
model.fit(Train_set_poly, Result)
prediction = model.predict(Test_set_poly)
r_sq = r2_score(Result_Test, prediction)
newStudentPoly = poly_features.fit_transform(newStudent)
predicted = model.predict(newStudentPoly)
print("Predetto : {}".format(predicted))
print("errore : {}".format(mean_squared_error(realValue,predicted)))
print("Coefficient of determination: ", r_sq)
print("Intercept: ", model.intercept_)
print("Slope:", model.coef_)


#   QUI COMINCIA LA PROVA CON TUTTI GLI ATTRIBUTI


predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set = []
test_set = []
train_result = []
test_result = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        train_result.append([predictiveAttributeDegree[i][2]])
    else:
        test_set.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        test_result.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8], predictiveAttributeNotDegree[i][9],
                          predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][17]])
        train_result.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8], predictiveAttributeNotDegree[i][9],
                          predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][17]])
        test_result.append([predictiveAttributeNotDegree[i][2]])

poly_reg_tot = LinearRegression()
poly_features = PolynomialFeatures(degree=2, include_bias=False)
Train_set_poly = poly_features.fit_transform(train_set)
Train_result_poly = poly_features.fit_transform(train_result)
Test_set_poly = poly_features.fit_transform(test_set)
poly_reg_tot.fit(Train_set_poly, Train_result_poly)
predictionAll = poly_reg_tot.predict(Test_set_poly)
r_sq = poly_reg_tot.score(Train_set_poly, Train_result_poly)
print("-----ALL ATTRIBUTE-----: Coefficient of determination: ", r_sq)
newStudent = [[633, 1355, 1, 1, 2013, 3, 2013, 92, 54, 9, 0]]
realValue = [42]
newStudentPoly = poly_features.fit_transform(newStudent)
print("Predetto : {}".format(predicted))
print("errore : {}".format(mean_squared_error(realValue,predicted)))