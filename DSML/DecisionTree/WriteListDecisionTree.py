"""
Questo file si occupa di effettuare la scissione degli studenti laureati da quelli non laureati(per qualsiasi motivo).
Si procede ad enumerare i valori possibili dei campi alfabetici, al fine di avere solamente valori numerici. Il dizionario delle
varie corrispondenze si trova nei file
Cod_School.txt, Mot_sta_stud.txt, sta_stud.txt, Tipo_mat.txt etc. Il significato di questi file è spiegato nel commento del
file DecisionTreeRegressor.py.
Successivamente i due "dataset" scissi verranno scritti su file al fine di renderli immutabili per ogni esecuzione del DecisionTree.
"""


import json
import numpy as np
import pandas as pd

student = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/ListDataset.txt", orient='records',dtype=True)
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
matr_dict = dict([(y,x+1) for x,y in enumerate((set(matr)))])
cf_dict = dict([(y,x+1) for x,y in enumerate((set(cf)))])
CDS_dict = dict([(y,x+1) for x,y in enumerate((set(CDS)))])
Tipo_Cds_dict = dict([(y,x+1) for x,y in enumerate((set(Tipo_Cds)))])
Cod_School_dict = dict([(y,x+1) for x,y in enumerate((set(Cod_School)))])
Tipo_mat_dict= dict([(y,x+1) for x,y in enumerate((set(tipo_mat)))])
mot_sta_stud_dict= dict([(y,x+1) for x,y in enumerate((set(mot_sta_stud)))])
sta_stud_dict= dict([(y,x+1) for x,y in enumerate((set(sta_stud)))])
predictiveAttribute = []
predictiveAttributeTemp = []
predictiveAttribute_not_degree = []
predictiveAttribute_not_degreeTemp = []
for k in matr_dict.keys():
    for i in range(len(matr)):
        if matr[i] == k:
            matr[i] = matr_dict.get(k)

for k in cf_dict.keys():
    for i in range(len(cf)):
        if cf[i] == k:
            cf[i] = cf_dict.get(k)

for k in CDS_dict.keys():
    for i in range(len(CDS)):
        if CDS[i] == k:
            CDS[i] = CDS_dict.get(k)

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

degree=sta_stud_dict.get("Laureato")
print(degree)
for i in range(0,len(matr)):
    if sta_stud[i] == degree:
        predictiveAttributeTemp = [int(matr[i]), int(cf[i]), int(targetValue[i]), int(secondo[i]), int(terzo[i]), int(tot[i]), int(CDS[i]), int(Tipo_Cds[i]), int(coorte[i]), int(anni_carriera[i]), int(anno_diploma[i]),
                        int(voto_diploma[i]), int(Cod_School[i]), int(tipo_mat[i]), int(anno_laurea[i]), int(voto_laurea[i]), int(erasmus[i]), int(tesi_estero[i]), int(mot_sta_stud[i]), int(sta_stud[i]), int(fc[i])]
        predictiveAttribute.append(predictiveAttributeTemp)

    elif sta_stud[i] != degree:
        predictiveAttribute_not_degreeTemp = [int(matr[i]), int(cf[i]), int(targetValue[i]), int(secondo[i]), int(terzo[i]), int(tot[i]), int(CDS[i]), int(Tipo_Cds[i]), int(coorte[i]), int(anni_carriera[i]), int(anno_diploma[i]),
                        int(voto_diploma[i]), int(Cod_School[i]), int(tipo_mat[i]), int(anno_laurea[i]), int(voto_laurea[i]), int(erasmus[i]), int(tesi_estero[i]), int(mot_sta_stud[i]), int(sta_stud[i]), int(fc[i])]
        predictiveAttribute_not_degree.append(predictiveAttribute_not_degreeTemp)
print(len(predictiveAttribute))
print(len(predictiveAttribute_not_degree))
with open('matr.txt', 'w') as file:
    file.write(json.dumps(matr_dict))

with open('cf.txt', 'w') as file:
    file.write(json.dumps(cf_dict))

with open('CDS.txt', 'w') as file:
    file.write(json.dumps(CDS_dict))

with open('Cod_school.txt', 'w') as file:
    file.write(json.dumps(Cod_School_dict))

with open('Tipo_mat.txt', 'w') as file:
    file.write(json.dumps(Tipo_mat_dict))

with open('Mot_sta_stud.txt', 'w') as file:
    file.write(json.dumps(mot_sta_stud_dict))

with open('sta_stud.txt', 'w') as file:
    file.write(json.dumps(sta_stud_dict))

with open('Tipo_CDS.txt', 'w') as file:
    file.write(json.dumps(Tipo_Cds_dict))

with open('predictiveDegree.txt', 'w') as file:
    file.write(json.dumps(predictiveAttribute))

with open('predictiveNotDegree.txt', 'w') as file:
    file.write(json.dumps(predictiveAttribute_not_degree))

