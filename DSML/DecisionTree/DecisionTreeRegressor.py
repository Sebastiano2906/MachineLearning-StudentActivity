"""
Implementazione di un DecisonTreeRegressor. In questo file si cerca di effettuare la regressione sul campo "1", ovvero
i CFU sostenuti al primo anno. Per farlo vengono presi in esame tutti gli attributi a disposizione. Siccome la regressione,
aspetta in input solamente valori numerici, ho mappato le stringhe in valori numerici. La tecnica utilizzata è stata quella
di utilizzare un dizionario, dove ogni elemento rappresenta la chiave e il valore è un semplice enumerazione. Eg.:
Tipo_mat_dict= dict([(y,x+1) for x,y in enumerate(sorted((set(tipo_mat))))]) con questa riga, vengono mappate tutte le maturità,
quindi "Scientifica" rappresenterà la chiave, e ad essa sarà associato un valore. Il risultato di questo mapping viene scritto nei file
Cod_School.txt, Mot_sta_stud.txt, sta_stud.txt, Tipo_mat.txt

Successivamente viene lanciato un decision tree, non prima di aver fatto cross-validation. Il punteggio ottenuto in termini di R2Score, è ottimo, 0.86.



EDIT: MI SONO ACCORTO CHE IL MODELLO SI ADATTAVA TROPPO ALLA DISTRIBUZIONE DEGLI STUDENTI. INFATTI LE PREDIZIONI ERANO MOLTO PIù
PRECISE QUANDO LO STUDENTE ERA UNO STUDENTE IN CORSO, E NON LAUREATO, PER FARE IN MODO DA SUDDIVIDERE GLI STUDENTI LAUREATI DA QUELI NON LAUREATI
HO CREATO IL FILE WriteListDecisionTree.py DOVE EFFETTUO LA SEPARAZIONE DEI DUE TIPI DI STUDENTI. LEGGO I DATI DAI RISPETTIVI FILE E GENERO TRAINSET E TESTSET
IN MODO DA MANTENERE UNA PERCENTUALE DI 80/20 DI COMPOSIZIONE. IL TRAIN SET QUINDI SARA COSTITUITO DALL 80% DEL DATASET TOTALE.

PER VISUALIZZARE L'ALBERO OUTPUT DI QUESTO ALGORITMO COPIARE IL CONTENUTO DEL FILE DecisionTree.txt e incollare tutto nella TextBox del sito http://webgraphviz.com/ .

Enjoy the happines.
"""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt
import numpy as np



predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set = []
test_set = []
train_result = []
test_result = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][13]])
        train_result.append([predictiveAttributeDegree[i][2]])
    else:
        test_set.append([predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][13]])
        test_result.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][13]])
        train_result.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][13]])
        test_result.append([predictiveAttributeNotDegree[i][2]])


regressor = DecisionTreeRegressor(random_state=0, min_samples_leaf=10)
print(cross_val_score(regressor, train_set[1:], train_result[1:], cv=10))
regressor.fit(train_set, train_result)
print(regressor.score(test_set, test_result))
prediction = []
for item in test_set:
    items = [[item[0], item[1]]]
    prediction.append(regressor.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
print(("MSE: {}".format(mean_squared_error(pred, test_result))))
print("Params: ", regressor.get_params())
print("Feature Importance: ", regressor.feature_importances_)

print("\n\n\n----------QUI INIZIA LA SEZIONE CON TUTTI GLI ATTRIBUTI---------- \n\n\n")
predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set_tot = []
test_set_tot = []
train_result_tot = []
test_result_tot = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        train_set_tot.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        train_result_tot.append([predictiveAttributeDegree[i][2]])
    else:
        test_set_tot.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][6],
                          predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8], predictiveAttributeDegree[i][9],
                          predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][17]])
        test_result_tot.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set_tot.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8], predictiveAttributeNotDegree[i][9],
                          predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][17]])
        train_result_tot.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set_tot.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][6],
                          predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8], predictiveAttributeNotDegree[i][9],
                          predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][17]])
        test_result_tot.append([predictiveAttributeNotDegree[i][2]])

regressorAllAttribute = DecisionTreeRegressor(random_state=0, min_samples_leaf=10)
print(cross_val_score(regressorAllAttribute, train_set_tot[1:], train_result_tot[1:], cv=10))
regressorAllAttribute.fit(train_set_tot, train_result_tot)
print(regressorAllAttribute.score(test_set_tot, test_result_tot))
prediction = []
for item in test_set_tot:
    items = [[item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10]]]
    prediction.append(regressorAllAttribute.predict(items))
pred = np.zeros(len(prediction))
predi = np.array(prediction)
for i in range(len(prediction)):
    pred[i] = predi[i][0]
print(("MSE: {}".format(mean_squared_error(pred, test_result_tot))))
print("---ALL ATTRIBUTE----: Params: ", regressorAllAttribute.get_params())
print("---ALL ATTRIBUTE----: Feature Importance: ", regressorAllAttribute.feature_importances_)
from sklearn.model_selection import train_test_split
import numpy as np
import pylab
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    pylab.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Errori su train set")
    pylab.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Errori su test set")
    pylab.legend(loc='upper right')
    pylab.show()


plot_learning_curves(regressorAllAttribute, train_set_tot, train_result_tot)

with open("DecisionTreeAllAttribute.txt", "w") as f:
    f = export_graphviz(regressorAllAttribute, out_file=f)


with open("DecisionTree.txt", "w") as f:
    f = export_graphviz(regressor, out_file=f)