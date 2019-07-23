import os
import pandas as pd
import csv
import json
DEGREE_PATH= "C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DatasetTriennali.csv"

def load_degree_data(degree_path=DEGREE_PATH):
    csv_path = os.path.join(degree_path, "C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DatasetTriennali.csv")
    return pd.read_csv(csv_path)

degree_db = load_degree_data()

laureati = [tuple(row) for row in csv.reader(open('DatasetTriennali.csv', 'r'))]

Maturità = []
Maturitàtemp = []
MaturitàDict= {}
SchoolDict = {}
nome_Maturità = " "
tipo_maturità = 0
for i in range(1, len(laureati)):
   nome_Maturità = laureati[i][13]
   cod_Scuola = laureati[i][12]
   tipo_maturità = int(abs(hash(nome_Maturità)) % (10 ** 8))
   nome_Scuola = int(abs(hash(cod_Scuola)) % (10 ** 8))
   MaturitàDict[tipo_maturità] = nome_Maturità
   SchoolDict[nome_Scuola] = cod_Scuola
   voto_diploma = int(laureati[i][11])
   CFU_primo = int(laureati[i][2])
   if CFU_primo != -1:
       if CFU_primo <= 60:
        Maturitàtemp = [[tipo_maturità, voto_diploma, CFU_primo, nome_Scuola]]
        Maturità.append(Maturitàtemp)


with open('ListStudent.txt', 'w') as file:
    file.write(json.dumps(Maturità))

with open('DictSchool.txt', 'w') as file:
    file.write(json.dumps(MaturitàDict))

with open('DictCodSchool.txt', 'w') as file:
    file.write(json.dumps(SchoolDict))