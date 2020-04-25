"""L'algoritmo K-neighbors neighbors (KNN) è un tipo di algoritmi di apprendimento automatico supervisionato.
svolge compiti di classificazione piuttosto complessi.
È un algoritmo di apprendimento pigro poiché non ha una fase di allenamento specializzata. Piuttosto,
utilizza tutti i dati per l'addestramento mentre classifica un nuovo punto dati o istanza.
KNN è un algoritmo di apprendimento non parametrico, il che significa che non assume nulla sui dati sottostanti."""
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

"""-----------------------ATTENZIONE NON RILANCIARE MI è SERVITO PER IL CSV----------------------"""

from sklearn.preprocessing import LabelEncoder
laureati = [tuple(row) for row in csv.reader(open('../DatasetTriennaliNuovo.csv', 'r'))] #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
Maturità = []
tip_mat= json.load(open('../Tipo_mat.txt')) #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
cod_school=json.load(open('../Cod_school.txt')) #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE


for i in range(1, len(laureati)):
<<<<<<< HEAD
    num=len(laureati[i][13])
    stringa=(laureati[i][13][1:num-1])
    tipo_maturità=tip_mat.get(stringa)
    voto_diploma = int(laureati[i][11])
    fuori_corso = int (laureati[i][20])
    """num = len(laureati[i][12])
       stringa = (laureati[i][12][1:num - 1])
       codice_mec=cod_school.get(stringa)
       print(codice_mec)"""
    if fuori_corso!=0:
      fuori_corso=1
=======
  #  num=len(laureati[i][13])
   # stringa=(laureati[i][13][1:num-1])
    #tipo_maturità=float(tip_mat.get(stringa))
    tipo_maturità= float(laureati[i][21])
    voto_diploma = float(laureati[i][11])
    fuori_corso = float (laureati[i][20])
    """num = len(laureati[i][12])
       stringa = (laureati[i][12][1:num - 1])
       codice_mec=cod_school.get(stringa)
       print(codice_mec)
    if fuori_corso!=0.0:
      fuori_corso=1.0
    """

>>>>>>> origin/master

    CFU_primo = int(laureati[i][2])
    if CFU_primo != -1:
       if CFU_primo <= 60:
        Maturitàtemp = [tipo_maturità, voto_diploma, CFU_primo, fuori_corso]# da classe a tipo
        Maturità.append(Maturitàtemp)


df = pd.DataFrame(data={"TipoMaturità, VotoDiploma,CFU1, AnniFuoriCorso": Maturità})
df.to_csv("./ListaStudentiModificata_ConseideraDatasetNuovo.csv", sep=',', index=False,)

