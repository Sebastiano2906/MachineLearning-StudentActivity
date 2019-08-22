from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import pandas as pd
import csv
import json
from sklearn.preprocessing import LabelEncoder



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
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
#dt={'TipoMaturità': 5,'VotoDiploma': 80,'CFU1':40, 'AnniFuoriCorso':0}
#kmeans.predict(dt)
print(kmeans.labels_)
print(centroids)
#print(df['TipoMaturità'])

#plt.scatter(df['CFU1'],df['AnniFuoriCorso'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:,0],centroids[:,1],c='red', s=50)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['VotoDiploma'],df['CFU1'],  df['AnniFuoriCorso'],c= kmeans.labels_.astype(float), s=50, alpha=0.5 )
ax.scatter(centroids[:,1],centroids[:,2],centroids[:,3],c='red', s=50)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
plt.show()

