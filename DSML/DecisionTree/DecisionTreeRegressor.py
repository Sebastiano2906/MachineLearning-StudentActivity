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
        train_set.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][3], predictiveAttributeDegree[i][4],
                          predictiveAttributeDegree[i][5], predictiveAttributeDegree[i][6], predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8],
                          predictiveAttributeDegree[i][9], predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][14], predictiveAttributeDegree[i][15], predictiveAttributeDegree[i][16],
                          predictiveAttributeDegree[i][17], predictiveAttributeDegree[i][18], predictiveAttributeDegree[i][19], predictiveAttributeDegree[i][20]])
        train_result.append([predictiveAttributeDegree[i][2]])
    else:
        test_set.append([predictiveAttributeDegree[i][0], predictiveAttributeDegree[i][1], predictiveAttributeDegree[i][3], predictiveAttributeDegree[i][4],
                          predictiveAttributeDegree[i][5], predictiveAttributeDegree[i][6], predictiveAttributeDegree[i][7], predictiveAttributeDegree[i][8],
                          predictiveAttributeDegree[i][9], predictiveAttributeDegree[i][10], predictiveAttributeDegree[i][11], predictiveAttributeDegree[i][12],
                          predictiveAttributeDegree[i][13], predictiveAttributeDegree[i][14], predictiveAttributeDegree[i][15], predictiveAttributeDegree[i][16],
                          predictiveAttributeDegree[i][17], predictiveAttributeDegree[i][18], predictiveAttributeDegree[i][19], predictiveAttributeDegree[i][20]])
        test_result.append([predictiveAttributeDegree[i][2]])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        train_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][3], predictiveAttributeNotDegree[i][4],
                          predictiveAttributeNotDegree[i][5], predictiveAttributeNotDegree[i][6], predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8],
                          predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][14], predictiveAttributeNotDegree[i][15], predictiveAttributeNotDegree[i][16],
                          predictiveAttributeNotDegree[i][17], predictiveAttributeNotDegree[i][18], predictiveAttributeNotDegree[i][19], predictiveAttributeNotDegree[i][20]])
        train_result.append([predictiveAttributeNotDegree[i][2]])
    else:
        test_set.append([predictiveAttributeNotDegree[i][0], predictiveAttributeNotDegree[i][1], predictiveAttributeNotDegree[i][3], predictiveAttributeNotDegree[i][4],
                          predictiveAttributeNotDegree[i][5], predictiveAttributeNotDegree[i][6], predictiveAttributeNotDegree[i][7], predictiveAttributeNotDegree[i][8],
                          predictiveAttributeNotDegree[i][9], predictiveAttributeNotDegree[i][10], predictiveAttributeNotDegree[i][11], predictiveAttributeNotDegree[i][12],
                          predictiveAttributeNotDegree[i][13], predictiveAttributeNotDegree[i][14], predictiveAttributeNotDegree[i][15], predictiveAttributeNotDegree[i][16],
                          predictiveAttributeNotDegree[i][17], predictiveAttributeNotDegree[i][18], predictiveAttributeNotDegree[i][19], predictiveAttributeNotDegree[i][20]])
        test_result.append([predictiveAttributeNotDegree[i][2]])

regressor = DecisionTreeRegressor(random_state=0, min_samples_leaf=10)
print(cross_val_score(regressor, train_set[1:], train_result[1:], cv=10))
regressor.fit(train_set, train_result)
print(regressor.score(test_set, test_result))
#              matr cf    2  3 tot cds tipoCds coorte annicarriera annodiploma votodip codschool tipoMat annolaur votolaur erasmus tesi mot_sta sta fc
newStudent = [[2933, 2928, 0, 0, 30, 1, 1, 2018, 1, 2018, 100, 200, 1, 0, 0, 0, 0, 1, 2, 0]]
real_value = [30]
predicted = regressor.predict(newStudent)
print("Predicted: ", predicted)
print("MSE: ", mean_squared_error(real_value, regressor.predict(newStudent)))
print("Params: ", regressor.get_params())
print("Feature Importance: ", regressor.feature_importances_)


with open("DecisionTree.txt", "w") as f:
    f = export_graphviz(regressor, out_file=f)