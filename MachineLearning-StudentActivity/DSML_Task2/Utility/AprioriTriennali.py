"""
Implementazione dell'algoritmo Apriori, visto a lezione e presenti slide di spiegazione sulla pagina e-learning.
Questo algoritmo cerca di trovare dei frequenti itemset, ossia ripetizioni frequenti di coppie di attributi. Per farlo
lavora sulle misure di confidenza e supporto -anche per queste misure c'è una spiegazione sulle slide-. Per provarlo,
non dovete fare altro che cambiare le misure di confidenza e supporto, settandole al valore che preferite. Ad esempio
settando min_suppor = 0.002 state dichiarando di volere tutti gli itemset che hanno un supporto minimo del 2%.
In questa implementazione i valori settati sono quelli di default per l'algoritmo, al di sotto di queste soglie non avrebbe
senso scendere.
L'algoritmo viene lanciato 2 volte, una volta solo sull'insieme di attributi {Tipo_Maturità,Voto_Diploma,CFU_Primo{ (eg.["Scientifica, 100, 54],
e successivamente sull'intero Dataset.

Risultati poco utili. Provare per credere.
"""
import pandas as pd
from apyori import apriori
import json
import csv
store_data = pd.read_csv('../DatasetStudenti.csv', header=None)#AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
records = [tuple(row) for row in csv.reader(open('../TotalStudent.csv', 'r'))] #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
print(records[:5])
dataset = []
for i in range(0, len(store_data)):
    dataset.append([str(store_data.values[i,j]) for j in range(0, 20)])
association_rules = apriori(records, min_support=0.002, min_confidence=0.1, min_lift=3, min_lenght= 2)
association_rules_dataset = apriori(dataset, min_support=0.0045, min_confidence=0.3, min_lift=3, min_lenght= 2)


for item in association_rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

print("=================================SULL'INTERO DATASET=================================")
for item in association_rules_dataset:
    with open('../OutputApriori.txt', 'w') as file: #AGGIUSTARE I PATH, IN MODO DA RAGGIUNGERE QUESTO FILE
        file.write("Rule: " + str(item[0]) + " -> " + str(item[1]) + "\n" + "Support: " + str(item[1]) + "\n" + "Confidence: " + str(item[2][0][2])  + "\n" +
                   "Lift: " + str(item[2][0][3]) + "\n" + "=====================================")
    # # first index of the inner list
    # # Contains base item and add item
    # pair = item[0]
    # items = [x for x in pair]
    # print("Rule: " + items[0] + " -> " + items[1])
    #
    # #second index of the inner list
    # print("Support: " + str(item[1]))
    #
    # #third index of the list located at 0th
    # #of the third index of the inner list
    #
    # print("Confidence: " + str(item[2][0][2]))
    # print("Lift: " + str(item[2][0][3]))
    # print("=====================================")