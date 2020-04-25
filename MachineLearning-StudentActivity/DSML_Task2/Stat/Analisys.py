import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import csv
import json
STUDENT_PATH="../DatasetTriennali.csv" #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
#STUDENT_PATH="../ListaStudenti4.csv"
#trasforma csv in DataFrame
student =pd.read_csv(STUDENT_PATH)


#tolgo ciò che non mi interessa e trasformo le stringhe in valori numerici
student=student.drop( ['Matr','CF','2','3','tot','CDS','Tipo_CDS','Coorte','Anni_Carriera','ANNO_DIPLOMA','CODICE_MECCANOGRAFICO','ANNO_ACCADEMICO_LAUREA','VOTO_LAUREA','Erasmus','TESI_ESTERO','STATO_STUDENTE','MOTIVO_STATO_STUDENTE','CLASSE'],axis=1)
le=LabelEncoder()
student['TIPO_MATURITA']=le.fit(student['TIPO_MATURITA']).transform(student['TIPO_MATURITA'])
#print("mostro il mio dataframe\n",student)





#mostra info del dataframe
print("mostro info dettagli struttura\n",student.info())
#mostra descrizione statistiche
print("mostro prima dati statistici\n",student.describe())
#conta per tipomaturità
print("contiamo per tipo di maturità\n",student["TIPO_MATURITA"].value_counts())
#matplotlib inline # only in a Jupyter notebook
"""
student.hist(bins=50, figsize=(20,15))
plt.show()


#dividi test e trein

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]



train_set, test_set = split_train_test(student, 0.2)
print(len(train_set), "train +", len(test_set), "test")

"""

#divido per id
import hashlib
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

student_with_id = student.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(student_with_id, 0.2, "index")

print(len(train_set), "train +", len(test_set), "test")

"""from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(student, test_size=0.2, random_state=42)"""

student2= student
#questa è la distribuzione
student.plot(kind="scatter", x="TIPO_MATURITA", y="VOTO_DIPLOMA")
#così vedi dove è più denso
student.plot(kind="scatter", x="TIPO_MATURITA", y="VOTO_DIPLOMA", alpha=0.1)
#plt.show()


corr_matrix = student2.corr()
print(corr_matrix['TIPO_MATURITA'].sort_values(ascending=False))

