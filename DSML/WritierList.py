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
nome_Maturità = " "
tipo_maturità = 0
for i in range(1, len(laureati)):
   nome_Maturità = laureati[i][13]
   tipo_maturità = int(abs(hash(nome_Maturità)) % (10 ** 8))
   voto_diploma = int(laureati[i][11])
   CFU_primo = int(laureati[i][2])
   if CFU_primo != -1:
    Maturitàtemp = [[tipo_maturità, voto_diploma, CFU_primo]]
    Maturità.append(Maturitàtemp)


with open('ListStudent.txt', 'w') as file:
    file.write(json.dumps(Maturità))
