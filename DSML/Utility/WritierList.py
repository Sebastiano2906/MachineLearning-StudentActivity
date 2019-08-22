"""
Questo codice serve per generare e rendere statici alcuni file che permettono di fare un mapping da stringhe ad interi.
Servito per la LinearRegression. Utile spunto per future necessità.
"""

import os
import pandas as pd
import csv
import json
DEGREE_PATH= "C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/Dataset/DatasetTriennali.csv"

def load_degree_data(degree_path=DEGREE_PATH):
    csv_path = os.path.join(degree_path, "C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/Dataset/DatasetTriennali.csv")
    return pd.read_csv(csv_path)

degree_db = load_degree_data()

laureati = [tuple(row) for row in csv.reader(open('C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/Dataset/DatasetTriennali.csv', 'r'))]
laureatiAug = [tuple(row) for row in csv.reader(open('C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/Dataset/DatasetTriennaliAugmented.csv', 'r'))]
Maturità = []
Maturitàtemp = []
MaturitàDict = {}
SchoolDict = {}
nome_Maturità = " "
tipo_maturità = 0
Total_Write = []
Total_Temp = []
classe_maturita = 0
for i in range(1, len(laureati)):
    nome_Maturità = laureati[i][13]
    cod_Scuola = laureati[i][12]
    tipo_maturità = int(abs(hash(nome_Maturità)) % (10 ** 8))
    nome_Scuola = int(abs(hash(cod_Scuola)) % (10 ** 8))
    classe_maturita = int(laureati[i][21])
    MaturitàDict[tipo_maturità] = nome_Maturità
    SchoolDict[nome_Scuola] = cod_Scuola
    voto_diploma = int(laureati[i][11])
    CFU_primo = int(laureati[i][2])
    if CFU_primo != -1:
       if CFU_primo <= 60:
        Maturitàtemp = [[classe_maturita, voto_diploma, CFU_primo]]
        Maturità.append(Maturitàtemp)
    Total_Temp = [nome_Maturità, voto_diploma, int(laureati[i][2]) + 1000]
    Total_Write.append(Total_Temp)

MaturitàtempAug = []
MaturitàAug = []
for i in range(1, len(laureatiAug)):
    voto_diploma = int(laureatiAug[i][11])
    CFU_primo = int(laureatiAug[i][2])
    if CFU_primo != -1:
       if CFU_primo <= 60:
        MaturitàtempAug = [[classe_maturita, voto_diploma, CFU_primo]]
        MaturitàAug.append(MaturitàtempAug)


with open('C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/ListStudent.txt', 'w') as file:
    file.write(json.dumps(Maturità))

with open('C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/ListStudentAug.txt', 'w') as file:
    file.write(json.dumps(MaturitàAug))

with open('C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictSchool.txt', 'w') as file:
    file.write(json.dumps(MaturitàDict))

with open('C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/DictCodSchool.txt', 'w') as file:
    file.write(json.dumps(SchoolDict))

df = pd.DataFrame(data={"Tipo_Maturita, Voto_Diploma, CFU_Primo": Total_Write})
df.to_csv("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/Dataset/TotalStudent.csv", sep=',', index=False,)