from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import pandas as pd
import csv
import json
from sklearn.preprocessing import LabelEncoder
import os


#Maturita = json.load(open(os.path.abspath("ListStudent3.txt"))) #ATTENZIONE AL PATH
Laureato = [tuple(row) for row in csv.reader(open(os.path.abspath("DatasetTriennaliNuovo.csv"), 'r'))]

tipo_maturita=[]
voto_diploma =[]
cfu_primo =[]
classe_CFU = []



for i in range(1, len(Laureato)):

   tipo_maturita.append(int(Laureato[i][21]))  # primo volore predittivo
   voto_diploma.append(int(Laureato[i][11]))  # secondo valore predittivo
   cfu_primo.append( int(Laureato[i][2]))

   #Trasformazione cfu in classi

   cfu = int(Laureato[i][2])

   if(cfu <= 15):

        classe_CFU.append(0)

   elif(cfu <= 30):

        classe_CFU.append(1)

   elif (cfu <= 45):

        classe_CFU.append(2)

   else:

        classe_CFU.append(3)



subset = {
    'TipoMaturita': tipo_maturita,
    'VotoDiploma': voto_diploma,
    }


df = DataFrame(subset, columns=['TipoMaturita', 'VotoDiploma'])


kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_

print("kmeans.labels_ = {0}".format(kmeans.labels_))
print("centroids = {0}".format(centroids))


DataFrameFinale= {
    'TipoMaturita': tipo_maturita,
    'VotoDiploma': voto_diploma,
    'CFU1':cfu_primo,
    'ClasseCFU' : classe_CFU,
    'Cluster':kmeans.labels_
    }


df_final = DataFrame(DataFrameFinale, columns=['TipoMaturita', 'VotoDiploma','CFU1','ClasseCFU','Cluster'])
#print ("STAMPA DATAFRAME FINALE \n {0}".format(df_final))


df_final.to_csv("./ListaStudentClasseCFU.csv", sep=',', index=False,)
ClusterFile = [tuple(row) for row in csv.reader(open("./ListaStudentClasseCFU.csv", 'r'))]

clusterCFU00 = 0
clusterCFU01 = 0
clusterCFU02 = 0
clusterCFU03 = 0
Totcluster0 = 0

clusterCFU10 = 0
clusterCFU11 = 0
clusterCFU12 = 0
clusterCFU13 = 0
Totcluster1 = 0

clusterCFU20 = 0
clusterCFU21 = 0
clusterCFU22 = 0
clusterCFU23 = 0
Totcluster2 = 0

clusterCFU30 = 0
clusterCFU31 = 0
clusterCFU32 = 0
clusterCFU33 = 0
Totcluster3 = 0

for i in range(1, len(ClusterFile)):

    #Conteggio cluster0
    if (int(ClusterFile[i][4]) == 0):

        Totcluster0 += 1

        if (int(ClusterFile[i][3]) == 0):

            clusterCFU00 +=1
        if (int(ClusterFile[i][3]) == 1):
            clusterCFU01 += 1

        if (int(ClusterFile[i][3]) == 2):
            clusterCFU02 += 1

        if (int(ClusterFile[i][3]) == 3):
            clusterCFU03 += 1
    #Conteggio cluster1
    elif(int(ClusterFile[i][4]) == 1):

        Totcluster1 += 1

        if (int(ClusterFile[i][3]) == 0):
            clusterCFU10 += 1
        if (int(ClusterFile[i][3]) == 1):
            clusterCFU11 += 1

        if (int(ClusterFile[i][3]) == 2):
            clusterCFU12 += 1

        if (int(ClusterFile[i][3]) == 3):
            clusterCFU13 += 1

    #COnteggio cluster2
    elif(int(ClusterFile[i][4]) == 2):

        Totcluster2 += 1

        if (int(ClusterFile[i][3]) == 0):
            clusterCFU20 += 1
        if (int(ClusterFile[i][3]) == 1):
            clusterCFU21 += 1

        if (int(ClusterFile[i][3]) == 2):
            clusterCFU22 += 1

        if (int(ClusterFile[i][3]) == 3):
            clusterCFU23 += 1

    #Conteggio cluster3
    else:

        Totcluster3 += 1

        if (int(ClusterFile[i][3]) == 0):
            clusterCFU30 += 1
        if (int(ClusterFile[i][3]) == 1):
            clusterCFU31 += 1

        if (int(ClusterFile[i][3]) == 2):
            clusterCFU32 += 1

        if (int(ClusterFile[i][3]) == 3):
            clusterCFU33 += 1


percentCluster00 = clusterCFU00 / Totcluster0
percentCluster01 = clusterCFU01 / Totcluster0
percentCluster02 = clusterCFU02 / Totcluster0
percentCluster03 = clusterCFU03 / Totcluster0

print("\n\nPercentuale di studenti nelle classi \n\t0 \t1 \t2 \t3  del cluster0: \n \t{0} , \t{1} , \t{2} , \t{3}"
      .format(percentCluster00, percentCluster01, percentCluster02, percentCluster03))


percentCluster10 = clusterCFU10 / Totcluster1
percentCluster11 = clusterCFU11 / Totcluster1
percentCluster12 = clusterCFU12 / Totcluster1
percentCluster13 = clusterCFU13 / Totcluster1

print("\n\nPercentuale di studenti nelle classi \n\t0 \t1 \t2 \t3  del cluster1: \n \t{0} , \t{1} , \t{2} , \t{3}"
      .format(percentCluster10, percentCluster11, percentCluster12, percentCluster13))


percentCluster20 = clusterCFU20 / Totcluster2
percentCluster21 = clusterCFU21 / Totcluster2
percentCluster22 = clusterCFU22 / Totcluster2
percentCluster23 = clusterCFU23 / Totcluster2

print("\n\nPercentuale di studenti nelle classi \n\t0 \t1 \t2 \t3  del cluster2: \n \t{0} , \t{1} , \t{2} , \t{3}"
      .format(percentCluster20, percentCluster21, percentCluster22, percentCluster23))


percentCluster30 = clusterCFU30 / Totcluster3
percentCluster31 = clusterCFU31 / Totcluster3
percentCluster32 = clusterCFU32 / Totcluster3
percentCluster33 = clusterCFU33 / Totcluster3

print("\n\nPercentuale di studenti nelle classi \n\t0 \t1 \t2 \t3  del cluster3: \n \t{0} , \t{1} , \t{2} , \t{3}"
      .format(percentCluster30, percentCluster31, percentCluster32, percentCluster33))


newStudent80 =[[9, 80]]
newStudent95 =[[11, 95]]
newStudent95bis =[[3, 95]]
newStudent65 =[[9, 65]]
newStudent85 =[[10, 85]]

print ("PREDIZIOONE STUDENTE80: {0}".format(kmeans.predict(newStudent80)))
print ("PREDIZIOONE STUDENTE95: {0}".format(kmeans.predict(newStudent95)))
print ("PREDIZIOONE STUDENTE95bis: {0}".format(kmeans.predict(newStudent95bis)))
print ("PREDIZIOONE STUDENTE65: {0}".format(kmeans.predict(newStudent65)))
print ("PREDIZIOONE STUDENTE85: {0}".format(kmeans.predict(newStudent85)))

