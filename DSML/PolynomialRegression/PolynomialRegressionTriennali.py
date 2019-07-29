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
poly_features = PolynomialFeatures(degree=2, include_bias=False)
Train_set, Result, Test_set, Result_Test = np.array(Train_set), np.array(Result), np.array(Test_set), np.array(Result_Test)
Train_set_poly = poly_features.fit_transform(Train_set)
model = LinearRegression()
model.fit(Train_set_poly, Result)
r_sq = model.score(Train_set_poly, Result)
print("Coefficient of determination: ", r_sq)
print("Intercept: ", model.intercept_)
print("Slope:", model.coef_)

maximum = 0.0
sw = 0
minus = 0.0
y_pred_pred = []
School = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictSchool.txt"))
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    predictiveTemp = [[tipo_maturità, voto_diploma]]
    predictiveTemp = np.array(predictiveTemp)
    predictiveTemp = poly_features.fit_transform(predictiveTemp)
    y_pred = model.predict(predictiveTemp)
    y_pred = float(y_pred)
    y_pred_pred.append(y_pred)
    if sw == 0:
        minus = y_pred
        min_school = tipo_maturità
        sw = 1
    elif minus > y_pred:
        minus = y_pred
        min_school = tipo_maturità
    if y_pred > maximum:
        maximum = y_pred
        max_school = tipo_maturità
max_school = str(max_school)
min_school = str(min_school)
name_max_school = School.get(max_school, None)
name_min_school = School.get(min_school, None)
print("Highest score: ", maximum, sep="\n")
print("Worst score: ", minus, sep="\n")
from sklearn.metrics import r2_score


print("R2Score : ", r2_score(Result_Test, y_pred_pred))