import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC as svc
from sklearn.model_selection import train_test_split
import csv


DEGREE_PATH="C:/Users/sebas/PycharmProjects/DSML/DatasetStudenti.csv"
def load_degree_data(degree_path=DEGREE_PATH):
    csv_path = os.path.join(degree_path, "C:/Users/sebas/PycharmProjects/DSML/DatasetStudenti.csv")
    return pd.read_csv(csv_path)


degree_db = load_degree_data()
"""degree_db.info()
print(degree_db.head(5))"""


laureati = [tuple(row) for row in csv.reader(open('DatasetStudenti.csv', 'r'))]
l2Degree = []
l2temp = []
lmDegree = []
lmtemp = []
for i in range(len(laureati)):
    if len(laureati[i]) == 21:
        if laureati[i][7] == 'L2':
            coorte = int(laureati[i][8])
            if coorte > 2007:
                years_carreer = int(laureati[i][9])
                year_lic = int(laureati[i][10])
                matr = int(laureati[i][0])
                CF = laureati[i][1]
                if laureati[i][2] != '?':
                    primo = int(laureati[i][2])
                else:
                    primo = -1
                if laureati[i][3] != '?':
                    secondo = int(laureati[i][3])
                else:
                    secondo = -1
                if laureati[i][4] != '?':
                    terzo = int(laureati[i][4])
                else:
                    terzo = -1
                if laureati[i][5] != '?':
                    tot = int(laureati[i][5])
                else:
                    tot = primo + secondo + terzo
                CDS = laureati[i][6]
                Tipo_Cds = 3
                if laureati[i][11] != '?':
                    vot_lic = int(laureati[i][11])
                else:
                    vot_lic = -1
                cod_school = laureati[i][12]
                school = laureati[i][13]
                if laureati[i][14] != '?':
                    year_of_degree = int(laureati[i][14])
                else:
                    year_of_degree = -1
                if laureati[i][15] != '?':
                    vot_degree = int(laureati[i][15])
                else:
                    vot_degree = -1
                if laureati[i][16] == '0':
                    erasmus = 0
                else:
                    erasmus = 1
                if laureati[i][17] == '0':
                    tesi_erasmus = 0
                else:
                    tesi_erasmus = 1
                state_stud = laureati[i][18]
                mot_stat_stud = laureati[i][19]
                if laureati[i][20] != '?':
                    if laureati[i][20] != '':
                        fc = int(laureati[i][20])
                else:
                    fc = -1
                l2temp = [matr, CF, primo, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc]
                l2Degree.append(l2temp)

for i in range(len(laureati)):
    if len(laureati[i]) == 21:
        if laureati[i][7] == 'LM':
            coorte = int(laureati[i][8])
            if coorte > 2007:
                years_carreer = int(laureati[i][9])
                matr = int(laureati[i][0])
                CF = laureati[i][1]
                if laureati[i][2] != '?':
                    primo = int(laureati[i][2])
                else:
                    primo = -1
                if laureati[i][3] != '?':
                    secondo = int(laureati[i][3])
                else:
                    secondo = -1
                if laureati[i][4] != '?':
                    terzo = int(laureati[i][4])
                else:
                    terzo = -1
                if laureati[i][5] != '?':
                    tot = int(laureati[i][5])
                else:
                    tot = primo + secondo + terzo
                CDS = laureati[i][6]
                Tipo_Cds = 2
                if laureati[i][11] != '?':
                    vot_lic = int(laureati[i][11])
                else:
                    vot_lic = -1
                cod_school = laureati[i][12]
                school = laureati[i][13]
                if laureati[i][14] != '?':
                    year_of_degree = int(laureati[i][14])
                else:
                    year_of_degree = -1
                if laureati[i][15] != '?':
                    vot_degree = int(laureati[i][15])
                else:
                    vot_degree = -1
                if laureati[i][16] == '0':
                    erasmus = 0
                else:
                    erasmus = 1
                if laureati[i][17] == '0':
                    tesi_erasmus = 0
                else:
                    tesi_erasmus = 1
                state_stud = laureati[i][18]
                mot_stat_stud = laureati[i][19]
                if laureati[i][20] != '?':
                    if laureati[i][20] != '':
                        fc = int(laureati[i][20])
                else:
                    fc = -1
                if laureati[i][10] != '?':
                    year_lic = int(laureati[i][10])
                else:
                    year_lic = -1
            lmtemp = [matr, CF, primo, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc]
            lmDegree.append(lmtemp)
print(len(l2Degree))
print(len(lmDegree))

df = pd.DataFrame(data={"Matr,CF,1,2,3,tot,CDS,Tipo_CDS,Coorte,Anni_Carriera,ANNO_DIPLOMA,VOTO_DIPLOMA,CODICE_MECCANOGRAFICO,TIPO_MATURITA,ANNO_ACCADEMICO_LAUREA,VOTO_LAUREA,Erasmus,TESI_ESTERO,STATO_STUDENTE,MOTIVO_STATO_STUDENTE,FC": lmDegree})
df.to_csv("./DatasetMagistrali.csv", sep=',', index=False,)
"""with open("DatasetMagistrali.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(lmDegree)
with open("DatasetTriennali.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(l2Degree)
#Funzione per convertire il Dataset in list

with open('DatasetStudenti.csv', 'r') as f:
    reader = csv.reader(f)
    laureati = list(reader)

"""

"""
#FUNZIONE PER SPLITTARE IL DATASET
"""
"""
def split_train_test(data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = train_test_split(degree_db,test_size=0.3, random_state=42)
print(len(train_set), "train +", len(test_set), "test")

corr_matrix = degree_db.corr()
print(corr_matrix)"""