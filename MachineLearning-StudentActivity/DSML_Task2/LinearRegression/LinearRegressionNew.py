import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #una libreria di visualizzazione dei dati
from sklearn.preprocessing import LabelEncoder #per codificare i valori stringa in campi numerici
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split #per suddividere la fase di addestramento con quella di test
from sklearn import linear_model #il modulo che include i vari tipi di regressione

Studenti = pd.read_csv("../DatasetTriennali.csv") #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE

print(Studenti.shape) #(2932 righe, 22 attributi)

#preparare dati, evitare null e sporcizia, ma dataset già pulito
#print(Studenti.isna().any())

#verifica correlazioni tra variabili
modelling_data= Studenti.copy()
modelling_data=modelling_data.drop( ['Matr','CF','2','3','tot','CDS','Tipo_CDS','Coorte','Anni_Carriera','ANNO_DIPLOMA','CODICE_MECCANOGRAFICO','ANNO_ACCADEMICO_LAUREA','VOTO_LAUREA','Erasmus','TESI_ESTERO','STATO_STUDENTE','MOTIVO_STATO_STUDENTE','CLASSE'],axis=1)
#print(len(modelling_data))
#print(modelling_data)

le=LabelEncoder()
modelling_data['TIPO_MATURITA']=le.fit(modelling_data['TIPO_MATURITA']).transform(modelling_data['TIPO_MATURITA'])
#abbiamo trasformato tipo maturità in un valore numerico
#print(modelling_data)

corr= modelling_data.corr()
mask=np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
f, ax = plt.subplots(figsize=(11,9))
cmap=sns.diverging_palette(0,150,as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink":.5})


#33% test 66% train
Train_set = []
Test_set = []
Result_Test = []
Total_Set = []
Tota_Result = []
Result = []

for i in range(0, len(modelling_data)):
    tipo_maturità = modelling_data['TIPO_MATURITA']#primo volore predittivo
    voto_diploma = modelling_data['VOTO_DIPLOMA']#secondo valore predittivo
    cfu_primo = modelling_data['1']#terzo valore predittivo

    TrainTemp = [tipo_maturità, voto_diploma, cfu_primo]
    Total_Temp = [tipo_maturità, voto_diploma,cfu_primo, modelling_data['FC']]
    Tota_Result.append(modelling_data['FC'])
    Result.append(modelling_data['FC'])
    Total_Set.append(TrainTemp)
    Train_set.append(TrainTemp)

modelling_data_new = modelling_data.copy()
modelling_data_new = Studenti.drop(['Matr','CF','1','2','3','tot','CDS','Tipo_CDS','Coorte','Anni_Carriera','ANNO_DIPLOMA','VOTO_DIPLOMA','CODICE_MECCANOGRAFICO','TIPO_MATURITA','ANNO_ACCADEMICO_LAUREA','VOTO_LAUREA','Erasmus','TESI_ESTERO','STATO_STUDENTE','MOTIVO_STATO_STUDENTE','CLASSE'],axis=1)
X_train, X_test, y_train, y_test= train_test_split(modelling_data, modelling_data_new, test_size=0.33, random_state=42)

reg= linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

reg.fit(X_train.values, y_train.values)
reg.coef_


print('Coefficiente di determinazione del dataset di addestramento:'+str(reg.score(X_train.values, y_train.values)))
print('Coefficiente di determinazione del dataset di test:'+str(reg.score(X_test.values, y_test.values)))

