from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import pandas as pd
import csv
import json
from sklearn.preprocessing import LabelEncoder



""" -------------------------genera file studenti----------------------------"""

DEGREE_PATH= "C:/Users/clara/PycharmProjects/prog2/DSML_Task2/Dataset/DatasetTriennali.csv"

def load_degree_data(degree_path=DEGREE_PATH):
    csv_path = os.path.join(degree_path, "C:/Users/clara/PycharmProjects/prog2/DSML_Task2/Dataset/DatasetTriennali.csv")
    return pd.read_csv(csv_path)

degree_db = load_degree_data()
laureati = [tuple(row) for row in csv.reader(open('C:/Users/clara/PycharmProjects/prog2/DSML_Task2/Dataset/DatasetTriennali.csv', 'r'))]
tip_mat= json.load(open('C:/Users/clara/PycharmProjects/prog2/DSML/FileGenerated/Tipo_mat.txt'))
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
    if fuori_corso!=0:
      fuori_corso=1

    CFU_primo = int(laureati[i][2])
    if CFU_primo != -1:
       if CFU_primo <= 60:
        Maturitàtemp = [[matricola_studente,tipo_maturità, voto_diploma, CFU_primo, fuori_corso]]# da classe a tipo
        Maturità.append(Maturitàtemp)





with open('C:/Users/clara/PycharmProjects/prog2/DSML_Task2/K-Means/ListStudent2.txt', 'w') as file:
    file.write(json.dumps(Maturità))



Maturità = json.load(open("C:/Users/clara/PycharmProjects/prog2/DSML_Task2/K-Means/ListStudent2.txt")) #ATTENZIONE AL PATH
tipo_maturità=[]
voto_diploma =[]
cfu_primo =[]
fc=[]
for i in range(0, len(Maturità)):
   tipo_maturità.append(int(Maturità[i][0][1]))  # primo volore predittivo
   voto_diploma.append(int(Maturità[i][0][2]))  # secondo valore predittivo
   cfu_primo.append( int(Maturità[i][0][3]))
   fc.append(int(Maturità[i][0][4]))

#for i in range(0,len(tipo_maturità)):
#   print(tipo_maturità[i])

Data = {
    'TipoMaturità': tipo_maturità,
    'VotoDiploma': voto_diploma,
    'CFU1':cfu_primo,
    'AnniFuoriCorso':fc
    }
le=LabelEncoder()



df = DataFrame(Data, columns=[ 'TipoMaturità', 'VotoDiploma','CFU1', 'AnniFuoriCorso'])

print(df)
kmeans = KMeans(n_clusters=6).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

