"""
Questo codice serve per generare e rendere statici alcuni file che permettono di fare un mapping da stringhe ad interi.
Servito per la LinearRegression. Utile spunto per future necessità.
"""

import os
import pandas as pd
import csv
import json
DEGREE_PATH= "../DatasetTriennali.csv" #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE

def load_degree_data(degree_path=DEGREE_PATH):
    csv_path = os.path.join(degree_path, "../DatasetTriennali.csv") #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
    return pd.read_csv(csv_path)

degree_db = load_degree_data()

laureati = [tuple(row) for row in csv.reader(open('../DatasetTriennali.csv', 'r'))] #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
#laureatiAug = [tuple(row) for row in csv.reader(open('../DatasetTriennaliAugmented.csv', 'r'))] #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
Maturità = []
Maturitàtemp = []
MaturitàDict= {}
SchoolDict = {}
nome_Maturità = " "
tipo_maturità = 0
Total_Write = []
Total_Temp = []
classe_maturita = 0
for i in range(1, len(laureati)):
    nome_Maturità = laureati[i][13]
    cod_Scuola = laureati[i][12]
    tipo_maturità = int(abs(hash(nome_Maturità)) % (10 ** 8)) #genera hash code per attributo tipo maturità e tipo nome scuola
    nome_Scuola = int(abs(hash(cod_Scuola)) % (10 ** 8))
    classe_maturita = int(laureati[i][21])
    MaturitàDict[tipo_maturità] = nome_Maturità
    SchoolDict[nome_Scuola] = cod_Scuola
    voto_diploma = int(laureati[i][11])
    fuori_corso = int (laureati[i][20])
    CFU_primo = int(laureati[i][2])
    if CFU_primo != -1:
       if CFU_primo <= 60:
        Maturitàtemp = [[classe_maturita, voto_diploma, CFU_primo, fuori_corso]]# da classe a tipo
        Maturità.append(Maturitàtemp)
    Total_Temp = [classe_maturita, voto_diploma, int(laureati[i][2]) + 1000, fuori_corso]
    Total_Write.append(Total_Temp)




with open('../ListStudent.txt', 'w') as file: #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
    file.write(json.dumps(Maturità))


with open('../DictSchool.txt', 'w') as file: #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
    file.write(json.dumps(MaturitàDict))

with open('../DictCodSchool.txt', 'w') as file: #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
    file.write(json.dumps(SchoolDict))

df = pd.DataFrame(data={"Tipo_Maturita, Voto_Diploma, CFU_Primo": Total_Write})
df.to_csv("../TotalStudent.csv", sep=',', index=False,)