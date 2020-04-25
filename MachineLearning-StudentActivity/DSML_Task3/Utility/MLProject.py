"""
Questo codice si occupa di fare una scrematura e pulizia dei dati. Vengono presi in considerazione
solo gli studenti iscritti dal 2012 in poi, che abbiamo il valore di 1 non nullo etc etc.
Una volta scremati i dati vengono scritti in un nuovo dataset. Le righe di codice commetante alla fine
sono o codice che serve e che è stato commentato per non riscrivere ad ogni esecuzione o tecniche alternative che sono
state sperimentate.
"""

import pandas as pd
import os
import csv

laureati = [tuple(row) for row in csv.reader(open('../DatasetStudenti.csv', 'r'))] #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
l2Degree = [['Matr', 'CF', '1', '2' , '3', 'tot', 'CDS', 'Tipo_CDS', 'Coorte', 'Anni_Carriera', 'ANNO_DIPLOMA', 'VOTO_DIPLOMA', 'CODICE_MECCANOGRAFICO', 'TIPO_MATURITA', 'ANNO_ACCADEMICO_LAUREA', 'VOTO_LAUREA', 'Erasmus', 'TESI_ESTERO', 'STATO_STUDENTE', 'MOTIVO_STATO_STUDENTE','FC']]
l2temp = []
lmDegree = []
lmtemp = []
for i in range(len(laureati)):
    if len(laureati[i]) == 21:
        if laureati[i][7] == 'L2':

            if laureati [i][13] == "Scientifica":
                coorte = int(laureati[i][8])
                if coorte > 2012:
                    years_carreer = int(laureati[i][9])
                    year_lic = int(laureati[i][10])
                    matr = int(laureati[i][0])
                    CF = laureati[i][1]
                    CF = CF[1:]
                    CF = int(CF)
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
                    CDS = CDS[1:len(CDS)-1]
                    CDS = int(CDS)
                    Tipo_Cds = 3
                    if laureati[i][11] != '?':
                        vot_lic = int(laureati[i][11])
                    else:
                        vot_lic = 0
                    cod_school = laureati[i][12]
                    school = laureati[i][13]

                    if laureati[i][14] != '?':
                        year_of_degree = int(laureati[i][14])
                    else:
                        year_of_degree = 0
                    if laureati[i][15] != '?':
                        vot_degree = int(laureati[i][15])
                    else:
                        vot_degree = 0
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
                        fc = 0
                    if fc < 5:
                        if vot_lic != 0:
                            if primo != -1:
                                l2temp = [matr, CF, primo, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc]
                                l2Degree.append(l2temp)


import json
with open('../ListDataset.txt', 'w') as file: #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
    file.write(json.dumps(l2Degree))


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