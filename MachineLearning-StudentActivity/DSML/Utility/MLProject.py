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

laureati = [tuple(row) for row in csv.reader(open('Dataset/DatasetStudenti.csv', 'r'))]
l2Degree = [['Matr', 'CF', '1', '2' , '3', 'tot', 'CDS', 'Tipo_CDS', 'Coorte', 'Anni_Carriera', 'ANNO_DIPLOMA', 'VOTO_DIPLOMA', 'CODICE_MECCANOGRAFICO', 'TIPO_MATURITA', 'ANNO_ACCADEMICO_LAUREA', 'VOTO_LAUREA', 'Erasmus', 'TESI_ESTERO', 'STATO_STUDENTE', 'MOTIVO_STATO_STUDENTE','FC', 'CLASSE']]
l2temp = []
lmDegree = []
lmtemp = []
for i in range(len(laureati)):
    if len(laureati[i]) == 21:
        if laureati[i][7] == 'L2':
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
                if school == "Scientifica":
                    classe = 3
                elif school == "Tecnico" or school == "Classica" or school == "Geometra":
                    classe = 2
                else:
                    classe = 1
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
                            l2temp = [matr, CF, primo, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc,classe]
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
                    primo = 0
                if laureati[i][3] != '?':
                    secondo = int(laureati[i][3])
                else:
                    secondo = 0
                if laureati[i][4] != '?':
                    terzo = int(laureati[i][4])
                else:
                    terzo = 0
                if laureati[i][5] != '?':
                    tot = int(laureati[i][5])
                else:
                    tot = primo + secondo + terzo
                CDS = laureati[i][6]
                Tipo_Cds = 2
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
                if laureati[i][10] != '?':
                    year_lic = int(laureati[i][10])
                else:
                    year_lic = 0
            if fc <= 5:
                lmtemp = [matr, CF, primo, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc]
                lmDegree.append(lmtemp)
print(len(l2Degree))
print(len(lmDegree))

#df = pd.DataFrame(data={"Matr,CF,1,2,3,tot,CDS,Tipo_CDS,Coorte,Anni_Carriera,ANNO_DIPLOMA,VOTO_DIPLOMA,CODICE_MECCANOGRAFICO,TIPO_MATURITA,ANNO_ACCADEMICO_LAUREA,VOTO_LAUREA,Erasmus,TESI_ESTERO,STATO_STUDENTE,MOTIVO_STATO_STUDENTE,FC,CLASSE": l2Degree})
#df.to_csv("./DatasetTriennali.csv", sep=',', index=False,)
import json
with open('ListDataset.txt', 'w') as file:
    file.write(json.dumps(l2Degree))


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

