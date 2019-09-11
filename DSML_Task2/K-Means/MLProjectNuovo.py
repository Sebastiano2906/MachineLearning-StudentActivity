import pandas as pd
import os
import csv
import json

laureati = [tuple(row) for row in csv.reader(open('C:/Users/clara/PycharmProjects/prog2/DSML_Task2/Dataset/DatasetStudenti.csv', 'r'))]
tip_mat= json.load(open('C:/Users/clara/PycharmProjects/prog2/DSML/FileGenerated/Tipo_mat.txt'))
cod_schoolset=json.load(open('C:/Users/clara/PycharmProjects/prog2/DSML/DecisionTree/Cod_school.txt'))
l2Degree = [
    ['Matr', 'CF', '1', '2', '3', 'tot', 'CDS', 'Tipo_CDS', 'Coorte', 'Anni_Carriera', 'ANNO_DIPLOMA', 'VOTO_DIPLOMA',
     'CODICE_MECCANOGRAFICO', 'TIPO_MATURITA', 'ANNO_ACCADEMICO_LAUREA', 'VOTO_LAUREA', 'Erasmus', 'TESI_ESTERO',
     'STATO_STUDENTE', 'MOTIVO_STATO_STUDENTE', 'FC', 'CLASSE']]
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
                CDS = CDS[1:len(CDS) - 1]
                CDS = int(CDS)
                Tipo_Cds = 3
                if laureati[i][11] != '?':
                    vot_lic = int(laureati[i][11])
                else:
                    vot_lic = 0

                num = len(laureati[i][12])
                stringa = (laureati[i][12][0:num])
                cod_school = cod_schoolset.get(stringa)
                """if school == "Scientifica":
                    classe = 3
                elif school == "Tecnico" or school == "Classica" or school == "Geometra":
                    classe = 2
                else:
                    classe = 1"""
                school =laureati[i][13]
                num = len(laureati[i][13])
                stringa = (laureati[i][13][0:num])
                print(stringa)
                classe = tip_mat.get(stringa)


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
                """se sei cessato e 180 allora 0, se sei attivo ma immatricolazione minore di 45 fuori corso 1, se sei cessato ma minore di 180 fuori corso 1"""
                if state_stud == "Cessato" and tot == 180:
                   fc=0
                if state_stud == "Cessato" and tot < 180:
                   fc=1
                if state_stud=="Attivo" and coorte <=2015:
                    fc=1
                if state_stud == "Attivo" and coorte > 2015:
                    fc=0
                if mot_stat_stud =="PASSAGGIO" and secondo==-1 or terzo==-1:
                    fc=1

                if vot_lic != 0:
                    if primo != -1 and primo<=60:
                        l2temp = [matr, CF, primo, secondo, terzo, tot, CDS, Tipo_Cds, coorte, years_carreer,
                                  year_lic, vot_lic, cod_school, school, year_of_degree, vot_degree, erasmus,
                                  tesi_erasmus, state_stud, mot_stat_stud, fc, classe]
                        l2Degree.append(l2temp)

with open("./DatasetTriennaliNuovo.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(l2Degree)