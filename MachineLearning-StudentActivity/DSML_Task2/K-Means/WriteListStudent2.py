import os
import pandas as pd
import csv
import json
from sklearn.preprocessing import LabelEncoder



""" -------------------------genera file studenti----------------------------"""

DEGREE_PATH= "../DatasetTriennali.csv" #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE

def load_degree_data(degree_path=DEGREE_PATH):
    csv_path = os.path.join(degree_path, "../DatasetTriennali.csv") #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
    return pd.read_csv(csv_path)

degree_db = load_degree_data()
laureati = [tuple(row) for row in csv.reader(open('../DatasetTriennali.csv', 'r'))] #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
tip_mat= json.load(open('../Tipo_mat.txt')) #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
cod_school=json.load(open('../Cod_school.txt')) #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
print(tip_mat)
Maturità = []
Maturitàtemp = []
MaturitàDict= {}
SchoolDict = {}
nome_Maturità = " "

Total_Write = []
Total_Temp = []
classe_maturita = 0
matricola =[]

for i in range(1, len(laureati)):
    matricola_studente=int(laureati[i][0])
   # classe_maturita = int(laureati[i][21])
    num=len(laureati[i][13])
    stringa=(laureati[i][13][1:num-1])
   # stringa="\""+stringa+"\""

   # print(tip_mat.get(stringa))
    tipo_maturità=tip_mat.get(stringa)
    voto_diploma = int(laureati[i][11])
    fuori_corso = int (laureati[i][20])
    """num = len(laureati[i][12])
       stringa = (laureati[i][12][1:num - 1])
       codice_mec=cod_school.get(stringa)
       print(codice_mec)
    if fuori_corso!=0:
      fuori_corso=1"""

    CFU_primo = int(laureati[i][2])
    if CFU_primo != -1:
       if CFU_primo <= 60:
        Maturitàtemp = [[matricola_studente,tipo_maturità, voto_diploma, CFU_primo, fuori_corso]]# da classe a tipo
        Maturità.append(Maturitàtemp)





with open('../ListStudent2.txt', 'w') as file: #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
    file.write(json.dumps(Maturità))
