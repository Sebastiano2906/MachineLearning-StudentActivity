"""
Implementazione di un DecisonTreeRegressor. In questo file si cerca di effettuare la regressione sul campo "1", ovvero
i CFU sostenuti al primo anno. Per farlo vengono presi in esame tutti gli attributi a disposizione. Siccome la regressione,
aspetta in input solamente valori numerici, ho mappato le stringhe in valori numerici. La tecnica utilizzata è stata quella
di utilizzare un dizionario, dove ogni elemento rappresenta la chiave e il valore è un semplice enumerazione. Eg.:
Tipo_mat_dict= dict([(y,x+1) for x,y in enumerate(sorted((set(tipo_mat))))]) con questa riga, vengono mappate tutte le maturità,
quindi "Scientifica" rappresenterà la chiave, e ad essa sarà associato un valore. Il risultato di questo mapping viene scritto nei file
Cod_School.txt, Mot_sta_stud.txt, sta_stud.txt, Tipo_mat.txt

Successivamente viene lanciato un decision tree, non prima di aver fatto cross-validation. Il punteggio ottenuto in termini di R2Score, è ottimo, 0.86.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import json


student = pd.read_json("ListDataset.txt", orient='records',dtype=True)
matr = np.array(student[0][1:])
cf = np.array(student[1][1:])
targetValue = np.array(student[2][1:])
secondo = np.array(student[3][1:])
terzo= np.array(student[4][1:])
tot = np.array(student[5][1:])
CDS = np.array(student[6][1:])
Tipo_Cds = np.array(student[7][1:])
coorte = np.array(student[8][1:])
anni_carriera = np.array(student[9][1:])
anno_diploma = np.array(student[10][1:])
voto_diploma = np.array(student[11][1:])
Cod_School = np.array(student[12][1:])
tipo_mat = np.array(student[13][1:])
anno_laurea = np.array(student[14][1:])
voto_laurea = np.array(student[15][1:])
erasmus = np.array(student[16][1:])
tesi_estero = np.array(student[17][1:])
mot_sta_stud = np.array(student[18][1:])
sta_stud = np.array(student[19][1:])
fc = np.array(student[20][1:])
classe = np.array(student[21][1:])
predictiveAttribute = []
Tipo_Cds_dict = dict([(y,x+1) for x,y in enumerate(sorted((set(Tipo_Cds))))])
Cod_School_dict = dict([(y,x+1) for x,y in enumerate(sorted((set(Cod_School))))])
Tipo_mat_dict= dict([(y,x+1) for x,y in enumerate(sorted((set(tipo_mat))))])
mot_sta_stud_dict= dict([(y,x+1) for x,y in enumerate(sorted((set(mot_sta_stud))))])
sta_stud_dict= dict([(y,x+1) for x,y in enumerate(sorted((set(sta_stud))))])
for k in Tipo_Cds_dict.keys():
    for i in range(len(Tipo_Cds)):
        if Tipo_Cds[i] == k:
            Tipo_Cds[i] = Tipo_Cds_dict.get(k)

for k in Cod_School_dict.keys():
    for i in range(len(Cod_School)):
        if Cod_School[i] == k:
            Cod_School[i] = Cod_School_dict.get(k)

for k in Tipo_mat_dict.keys():
    for i in range(len(tipo_mat)):
        if tipo_mat[i] == k:
            tipo_mat[i] = Tipo_mat_dict.get(k)

for k in mot_sta_stud_dict.keys():
    for i in range(len(mot_sta_stud)):
        if mot_sta_stud[i] == k:
            mot_sta_stud[i] = mot_sta_stud_dict.get(k)

for k in sta_stud_dict.keys():
    for i in range(len(sta_stud)):
        if sta_stud[i] == k:
            sta_stud[i] = sta_stud_dict.get(k)

for i in range(0,len(matr)):
        predictiveAttribute.append([matr[i], cf[i], secondo[i], terzo[i], tot[i], CDS[i], Tipo_Cds[i], coorte[i], anni_carriera[i], anno_diploma[i],
                        voto_diploma[i], Cod_School[i], tipo_mat[i], anno_laurea[i], voto_laurea[i], erasmus[i], tesi_estero[i], mot_sta_stud[i], sta_stud[i], fc[i]])

with open('Cod_school.txt', 'w') as file:
    file.write(json.dumps(Cod_School_dict))

with open('Tipo_mat.txt', 'w') as file:
    file.write(json.dumps(Tipo_mat_dict))

with open('Mot_sta_stud.txt', 'w') as file:
    file.write(json.dumps(mot_sta_stud_dict))

with open('sta_stud.txt', 'w') as file:
    file.write(json.dumps(sta_stud_dict))

regressor = DecisionTreeRegressor(random_state=0)
print(cross_val_score(regressor, predictiveAttribute[1:], targetValue[1:], cv=10))
regressor.fit(predictiveAttribute[:2346], targetValue[:2346])
print(regressor.score(predictiveAttribute[2347:], targetValue[2347:]))
newStudent = [[2934, 100, 40, 20, 100, 0o6126, 1, 2015, 3, 2015, 100, 200, 7, 2018, 108, 0, 0, 2, 6, 0]]
print("Predicted: ", regressor.predict(newStudent))
print("Params: ", regressor.get_params())
print("Feature Importance: ", regressor.feature_importances_)