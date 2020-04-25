import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import csv
import json

from sklearn.preprocessing import LabelEncoder
triennali="../ListaStudentiModificata_ConseideraDatasetNuovo.csv" #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
# Assign colum names to the dataset
names = [ 'TipoMaturità', 'VotoDiploma','CFU1', 'AnniFuoriCorso']

# Read dataset to pandas dataframe
dataset = pd.read_csv(triennali)
print(dataset)

#Il prossimo passo è dividere il nostro set di dati nei suoi attributi ed etichette.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

print("vediamo",X)
print("la tua y",y)

#Per creare training e divisioni di test, eseguire il seguente script:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Non esiste un valore ideale per K ed è selezionato dopo il test e la valutazione, tuttavia per iniziare, 5 sembra essere il valore più comunemente usato per l'algoritmo KNN.
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
arra=np.array([11,72,18])
arra = arra.reshape(1,-1)
classifier.fit(X_train, y_train)
print("predict",classifier.predict(arra))

#Il passo finale è fare previsioni sui nostri dati di test. Per fare ciò, eseguire il seguente script:
y_pred = classifier.predict(X_test)


#Per valutare un algoritmo, matrice di confusione, precisione, richiamo e punteggio f1 sono le metriche più comunemente utilizzate.
# I metodi confusion_matrixe classification_reportdel sklearn.metricspossono essere usati per calcolare queste metriche.
# Dai un'occhiata al seguente script:
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


"""Nella sezione di addestramento e previsione abbiamo detto che non c'è modo di sapere in anticipo quale valore di K produce i migliori risultati al primo tentativo. Abbiamo scelto casualmente 5 come valore K e ci risulta solo una precisione del 100%.

Un modo per aiutarti a trovare il valore migliore di K è quello di tracciare il grafico del valore K e il tasso di errore corrispondente per il set di dati.

In questa sezione, tracciamo l'errore medio per i valori previsti del set di test per tutti i valori K compresi tra 1 e 40.

Per fare ciò, calcoliamo prima la media dell'errore per tutti i valori previsti in cui K è compreso tra 1 e 40. Esegui il seguente script:"""


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
"""Lo script sopra esegue un ciclo da 1 a 40. In ogni iterazione viene calcolato l'errore medio per i valori previsti del set di test e il risultato viene aggiunto errorall'elenco.

Il prossimo passo è tracciare i errorvalori rispetto ai valori K. Eseguire il seguente script per creare la trama:"""

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

plt.show()
"""
precision: per tutte le istanze classificate come positive, quale percentuale era corretta?”

recall: è la capacità di un classificatore di trovare tutte le istanze positive.
Per ogni classe è definito come il rapporto tra i veri positivi e la somma dei veri positivi e dei falsi negativi.
Detto in altri termini, “per tutte le istanze che erano effettivamente positive, quale percentuale è stata classificata correttamente?”


il punteggio f1: Il punteggio F1 è una media armonica ponderata delle due precedenti metriche in modo tale che il 
punteggio migliore sia 1,0 e il peggiore sia 0,0. 
Come regola generale, la media ponderata di F1 dovrebbe essere utilizzata per confrontare i modelli di classificatore, 
non la precisione globale.

Supporto: Il supporto è il numero di occorrenze effettive della classe nel set di dati specificato.
Il supporto squilibrato nei dati di addestramento può indicare debolezze strutturali nei punteggi riportati
del classificatore e potrebbe indicare la necessità di campionamento stratificato o ribilanciamento. 
Il supporto non cambia tra i modelli, ma invece diagnostica il processo di valutazione."""