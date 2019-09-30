from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import pandas as pd
import csv
import json
from sklearn.preprocessing import LabelEncoder


Maturità = json.load(open("C:/Users/clara/PycharmProjects/prog2/DSML_Task2/K-Means/ListStudent2.txt")) #ATTENZIONE AL PATH
Laureato = [tuple(row) for row in csv.reader(open("C:/Users/clara/PycharmProjects/prog2/DSML_Task2/K-Means/DatasetTriennaliNuovo.csv", 'r'))]
tipo_maturità=[]
voto_diploma =[]
cfu_primo =[]
mat_student=[]
fc=[]
for i in range(1, len(Laureato)):
   #mat_student.append(int(Maturità[i][0][0]))
   tipo_maturità.append(int(Laureato[i][21]))  # primo volore predittivo
   voto_diploma.append(int(Laureato[i][11]))  # secondo valore predittivo
   cfu_primo.append( int(Laureato[i][2]))
   fc.append(int(Laureato[i][20]))

#for i in range(0,len(tipo_maturità)):
#   print(tipo_maturità[i])

Data = {
   # 'Matricola':mat_student,
    'TipoMaturità': tipo_maturità,
    'VotoDiploma': voto_diploma,
    'CFU1':cfu_primo,
    'AnniFuoriCorso':fc
    }
le=LabelEncoder()



df = DataFrame(Data, columns=['TipoMaturità', 'VotoDiploma','CFU1', 'AnniFuoriCorso'])
df2=df
df2=df2.drop(columns="AnniFuoriCorso")
#print(df2)

kmeans = KMeans(n_clusters=2).fit(df2)
centroids = kmeans.cluster_centers_




print(kmeans.labels_)
print(centroids)
#print(df['TipoMaturità'])
label=np.array(kmeans.labels_)



DataFrameFinale= {
   # 'Matricola':mat_student,
    'TipoMaturità': tipo_maturità,
    'VotoDiploma': voto_diploma,
    'CFU1':cfu_primo,
    'AnniFuoriCorso':fc,
    'Cluster':kmeans.labels_
    }
le=LabelEncoder()



df_final = DataFrame(DataFrameFinale, columns=['TipoMaturità', 'VotoDiploma','CFU1', 'AnniFuoriCorso','Cluster'])

#print("-------------------------------------PROVA---------------------------------------------")
#print(df_final)

dt=[[11,70,50]]

print(kmeans.predict(dt))
count0=0
countFc0 =0
count1 =0
countFc1 =0
count2 =0
countFc2 =0
count3 = 0
countFc3 =0

media0=0
media1=0
media2=0
media3=0








"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(df2['TipoMaturità'])
y = np.array(df2['VotoDiploma'])
z = np.array(df2['CFU1'])
ax.set_xlabel('TipoMaturità')
ax.set_ylabel('VotoDiploma')
ax.set_zlabel('CFU1')

ax.scatter(x,y,z, marker="s", c=kmeans.labels_.astype(float), s=50, cmap="RdBu", alpha=0.5 )

plt.show()

"""

df_final.to_csv("./ListaStudentClusterStatoStudenti.csv", sep=',', index=False,)
ClusterFile = [tuple(row) for row in csv.reader(open("./ListaStudentClusterStatoStudenti.csv", 'r'))]



for i in range(1, len(ClusterFile)):

    if int(ClusterFile[i][4]) == 0:
        count0 = count0 +1
        if int(ClusterFile[i][3]) > 0:
            countFc0 = countFc0 + 1
    if int(ClusterFile[i][4]) == 1:
        count1 = count1 +1
        if int(ClusterFile[i][3]) > 0:
            countFc1 = countFc1 +1

""" if int(ClusterFile[i][4]) == 2:
       count2 = count2 +1
       if int(ClusterFile[i][3]) > 0:
           countFc2 = countFc2 +1
   if int(ClusterFile[i][4]) == 3:
       count3 = count3 +1
       if int(ClusterFile[i][3]) > 0:
        countFc3 = countFc3 + 1"""

media0=countFc0/count0
media1=countFc1/count1
#media2=countFc2/count2
#media3=countFc3/count3

print("numero NON fuori corso :",countFc0)
print("numero fuori corso :",countFc1)
print("numero NON fuori corso PREDETTO :",count0)
print("numero fuori corso PREDETTO :",count1)
print("Cluster media fc 0 1 2 3 :")
print(media0,media1)

print("cluster0", count0)