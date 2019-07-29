"""
In questo algoritmo viene effettuata la Data Augmentation per cercare di sopperire alla scarsità di dati. La tecnica utilizzata
è quella di inserire per ogni studente 4 nuove tuple, ognuna contiene rispettivamente gli stessi attributi di quella prese in analisi,
ad eccezione del numero di CFU, del CF, e viene aggiunta anche la classe delle maturità (Determinata nel file ClassiMaturità.py).
Al campo CFU vengono sommati i seguenti valori (-9,-6,6,9) ogni nuovo valore rappresenterà una nuova tupla. Per distinguere le tuple aggiunte
da quelle reali il campo CF sarà "AUG".
Questo dataset successivamente viene scitto in un csv : /Dataset/DatasetTriennaliAugmented.csv.
Risultati pessimi quando si prende in considerazione questo dataset.

Provare per credere!
"""
import csv

laureati = [tuple(row) for row in csv.reader(open('Dataset/DatasetStudenti.csv', 'r'))]
l2Degree = [['Matr', 'CF', '1', '2' , '3', 'tot', 'CDS', 'Tipo_CDS', 'Coorte', 'Anni_Carriera', 'ANNO_DIPLOMA', 'VOTO_DIPLOMA', 'CODICE_MECCANOGRAFICO', 'TIPO_MATURITA', 'ANNO_ACCADEMICO_LAUREA', 'VOTO_LAUREA', 'Erasmus', 'TESI_ESTERO', 'STATO_STUDENTE', 'MOTIVO_STATO_STUDENTE','FC', 'CLASSE']]
l2temp = []
l2tempm9 = []
l2tempm6 = []
l2tempp6 = []
l2tempp9 = []
lmDegree = []
lmtemp = []
primop6 = 0
primop9 = 0
primom6 = 0
primom9 = 0
for i in range(len(laureati)):
    if len(laureati[i]) == 21:
        if laureati[i][7] == 'L2':
            coorte = int(laureati[i][8])
            if coorte > 2012:
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
                if 51 >= primo and primo != -1 and primo != 0 and primo >= 9:
                    primom9 = primo - 9
                    primom6 = primo - 6
                    primop6 = primo + 6
                    primop9 = primo + 9
                if fc < 5:
                    if vot_lic != 0:
                        if primo != -1:
                            l2temp = [matr, CF, primo, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc,classe]
                            l2tempm9 = [matr + len(laureati), "Aug", primom9, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc,classe]
                            l2tempm6 = [matr + len(laureati), "Aug", primom6, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc,classe]
                            l2tempp6 = [matr + len(laureati), "Aug", primop6, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc,classe]
                            l2tempp9 = [matr + len(laureati), "Aug", primop9, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer, year_lic, vot_lic, cod_school, school,year_of_degree, vot_degree, erasmus, tesi_erasmus, state_stud,mot_stat_stud,fc,classe]
                            l2Degree.append(l2temp)
                            l2Degree.append(l2tempm9)
                            l2Degree.append(l2tempm6)
                            l2Degree.append(l2tempp6)
                            l2Degree.append(l2tempp9)

def sortSecond(val):
    return val[0]

import json
with open('ListDatasetAug.txt', 'w') as file:
    file.write(json.dumps(l2Degree))
"""l2Degree.sort(key=sortSecond)
df = pd.DataFrame(data={"Matr,CF,1,2,3,tot,CDS,Tipo_CDS,Coorte,Anni_Carriera,ANNO_DIPLOMA,VOTO_DIPLOMA,CODICE_MECCANOGRAFICO,TIPO_MATURITA,ANNO_ACCADEMICO_LAUREA,VOTO_LAUREA,Erasmus,TESI_ESTERO,STATO_STUDENTE,MOTIVO_STATO_STUDENTE,FC,CLASSE": l2Degree})
df.to_csv("./DatasetTriennaliAugmented.csv", sep=',', index=False,)"""