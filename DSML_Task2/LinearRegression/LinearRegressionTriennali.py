"""
Algoritmo di linear regression.
Vengono dati in input i campi "Tipo_Mat" , "Voto_Diploma" e cfu_primo (letti dal file ListStudent.txt)
e si cerca di estrapolare un'equazione in grado di predirre se lo studente andrà o meno fuori corso.


I risultati sono pessimi poichè:
1. La quantità di dati a disposizione è minima
2. Gli attributi predittivi presi in esame sono altamente scorrelati fra di loro




Il coefficiente di determinazione , indicato come 𝑅², indica quale quantità di variazione in 𝑦 può essere spiegata
 dalla dipendenza da 𝐱 usando il particolare modello di regressione.
 Più grande 𝑅² indica un adattamento migliore e significa che il modello può spiegare meglio la variazione dell'output con input diversi.

Il valore 𝑅² = 1 corrisponde a SSR = 0, ovvero alla misura perfetta poiché i valori delle risposte previste ed effettive
 si adattano completamente l'uno all'altro.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import json
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd


Predictive_Value = []
predictiveTemp = []
Result = []
Maturità = json.load(open("C:/Users/clara/PycharmProjects/prog2/DSML_Task2/FileGenerated/ListStudent.txt")) #ATTENZIONE AL PATH

#Dict=json.load(open("C:/Users/clara/PycharmProjects/prog2/DSML_Task2/FileGenerated/DictSchool.txt"))

#print(Dict)
#divido io dataset 80 train e 20 test
Train_size = int((len(Maturità) / 100) * 80)
Test_size = int((len(Maturità)/100) * 20)
Train_set = []
Test_set = []
Result_Test = []
Total_Set = []
Tota_Result = []
# Total_Write = []
# Total_Temp = []
count =0
for i in range(0, len(Maturità)):
    tipo_maturità = int(Maturità[i][0][0])#primo volore predittivo
    voto_diploma = int(Maturità[i][0][1])#secondo valore predittivo
    cfu_primo = int(Maturità[i][0][2])#terzo valore predittivo
    count=count+1
    TrainTemp = [tipo_maturità, voto_diploma, cfu_primo]
    Total_Temp = [tipo_maturità, voto_diploma,cfu_primo, int(Maturità[i][0][3])]
    Tota_Result.append(int(Maturità[i][0][3]))
    Result.append(int(Maturità[i][0][3]))
    Total_Set.append(TrainTemp)
    Train_set.append(TrainTemp)
    #Total_Write.append(Total_Temp)

print(count)
# df = pd.DataFrame(data={"Tipo_Maturita, Voto_Diploma, CFU_Primo": Total_Write})
# df.to_csv("./TotalStudent.csv", sep=',', index=False,)
Train_set, Test_set, Result,  Result_Test= train_test_split(Train_set, Result, test_size=0.3)
Train_set, Result = np.array(Train_set), np.array(Result)
Train_set, Result = np.array(Train_set), np.array(Result)

"""   
Quando si implementa la regressione lineare di una variabile dipendente 𝑦 sull'insieme di variabili indipendenti 
𝐱 = (𝑥₁,…, 𝑥ᵣ), dove 𝑟 è il numero di predittori, si assume una relazione lineare tra 𝑦 e 𝐱: 𝑦 = 𝛽₀ + 𝛽₁𝑥₁ + ⋯ + 𝛽ᵣ𝑥ᵣ + 𝜀.
 Questa equazione è l' equazione di regressione . 
 𝛽₀, 𝛽₁,…, 𝛽ᵣ sono i coefficienti di regressione e 𝜀 è l' errore casuale .
 
 La regressione lineare calcola gli stimatori dei coefficienti di regressione o semplicemente i pesi previsti , indicati con 𝑏₀, 𝑏₁, ..., 𝑏ᵣ. 
 Definiscono la funzione di regressione stimata 𝑓 (𝐱) = 𝑏₀ + 𝑏₁𝑥₁ + ⋯ + 𝑏ᵣ𝑥ᵣ. 
 Questa funzione dovrebbe catturare sufficientemente bene le dipendenze tra input e output.
 
 
 La risposta stimata o prevista , 𝑓 (𝐱ᵢ), per ogni osservazione 𝑖 = 1,…, 𝑛, deve essere il più vicino possibile alla risposta effettiva corrispondente 𝑦ᵢ.
  Le differenze 𝑦ᵢ - 𝑓 (𝐱ᵢ) per tutte le osservazioni 𝑖 = 1,…, 𝑛, sono chiamate residui . 
  La regressione riguarda la determinazione dei pesi migliori previsti , ovvero i pesi corrispondenti ai residui più piccoli.
  
  Per ottenere i pesi migliori, di solito si minimizza la somma dei residui quadrati (SSR) per tutte le osservazioni 𝑖 = 1,…, 𝑛: SSR = Σᵢ (𝑦ᵢ - 𝑓 (𝐱ᵢ)) ². 
  Questo approccio è chiamato il metodo dei minimi quadrati ordinari .
  
  Il coefficiente di determinazione , indicato come 𝑅², indica quale quantità di variazione in 𝑦 può essere spiegata dalla dipendenza da 𝐱 
  usando il particolare modello di regressione. Più grande 𝑅² indica un adattamento migliore e significa che il modello può spiegare meglio la 
  variazione dell'output con input diversi.

Il valore 𝑅² = 1 corrisponde a SSR = 0, ovvero alla misura perfetta poiché i valori delle risposte previste ed 
effettive si adattano completamente l'uno all'altro.
 Il tuo obiettivo è calcolare i valori ottimali dei pesi previsti 𝑏₀ e 𝑏₁ che minimizzano l'SSR e determinano la funzione di regressione stimata. Il valore di 𝑏₀, 
 chiamato anche intercetta , mostra il punto in cui la linea di regressione stimata attraversa l'asse 𝑦. 
"""
model = LinearRegression()
model.fit(Train_set, Result)
r_sq = model.score(Train_set, Result)
print("Coefficient of determination: ", r_sq)
print("intercept: ", model.intercept_)
print("slope: ", model.coef_)
maximum = 0.0
sw = 0
minus = 0.0
y_pred_pred = []
for item in Test_set:
    tipo_maturità = item[0]
    voto_diploma = int(item[1])
    cfu_primo =int(item[2])
    predictiveTemp = [[tipo_maturità, voto_diploma, cfu_primo]]
    predictiveTemp = np.array(predictiveTemp)
    y_pred = model.predict(predictiveTemp)
    y_pred = float(y_pred)
    y_pred_pred.append(y_pred)
    if y_pred > 0:
        if sw == 0:
            minus = y_pred
            min_school = tipo_maturità
            sw = 1
        elif minus > y_pred:
            minus = y_pred
            min_school = tipo_maturità
        if y_pred > maximum:
            maximum = y_pred
            max_school = tipo_maturità

print("Highest score: ", maximum, "of type ",tipo_maturità, sep="\n")
print("Worst score: ", minus, "of school ", tipo_maturità, sep="\n")


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


plot_learning_curves(model, Total_Set, Tota_Result)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("R2Score : ", r2_score(Result_Test, y_pred_pred))
print("MSE : ", mean_squared_error(Result_Test, y_pred_pred))