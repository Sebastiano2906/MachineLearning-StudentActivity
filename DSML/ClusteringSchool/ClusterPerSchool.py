import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")

School = []
Students = []
for i in range(0, len(predictiveAttributeDegree)):
    if predictiveAttributeDegree[i][13] == 9:
        cfu = predictiveAttributeDegree[i][2]
        if cfu <= 15:
            predictiveAttributeDegree[i][2] = 0
        elif cfu <= 30:
            predictiveAttributeDegree[i][2] = 1
        elif cfu <= 45:
            predictiveAttributeDegree[i][2] = 2
        else:
            predictiveAttributeDegree[i][2] = 3
        scuola = predictiveAttributeDegree[i][12] #### <------- STO PRENDENDO IL CODICE MECCANOGRAFICO
        if scuola not in School:
            School.append(scuola)

for i in range(0, len(predictiveAttributeNotDegree)):
    if predictiveAttributeNotDegree[i][13] == 9:
        cfu = predictiveAttributeNotDegree[i][2]
        if cfu <= 15:
            predictiveAttributeNotDegree[i][2] = 0
        elif cfu <= 30:
            predictiveAttributeNotDegree[i][2]= 1
        elif cfu <= 45:
            predictiveAttributeNotDegree[i][2] = 2
        else:
            predictiveAttributeNotDegree[i][2] = 3
        scuola = predictiveAttributeNotDegree[i][12]
        if scuola not in School:
            School.append(scuola)

#kmeans = []
#centroids = []
voto = []
cfu_primo = []
count=0
for i in range(0, len(School)):
    for j in range(0, len(predictiveAttributeDegree)):
        if predictiveAttributeDegree[j][12] == School[i]:
            count=count+1
            voto.append(predictiveAttributeDegree[j][11])
            cfu_primo.append(predictiveAttributeDegree[j][2])
    for j in range(0, len(predictiveAttributeNotDegree)):
        if predictiveAttributeNotDegree[j][12] == School[i]:
            voto.append(predictiveAttributeNotDegree[j][11])
            cfu_primo.append(predictiveAttributeNotDegree[j][2])
            count=count+1
    kmeans = KMeans(n_clusters=4)
    voto = np.array(voto)
    voto=voto.reshape(-1,1)
    kmeans.fit(voto)
    centroids= kmeans.cluster_centers_
    voto = []
    count=0
    cfu_primo = []


print("Centroide 1 : " , centroids)

""" 
 data={
        "Voto":voto
    }
    df = DataFrame(data, columns=['Voto'])
    kmeans = KMeans(n_clusters=4).fit(df)
    centroids = kmeans.cluster_centers_
"""