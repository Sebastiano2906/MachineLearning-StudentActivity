"""
Questo algoritmo mappa le varie maturità nelle classi 1, 2 e 3. Per farlo viene fatta una media dei cfu fatti al primo
anno per ogni studente di un tipo di maturità. Ad esempio, gli studenti provenienti dallo Scientifico in media
al primo anno sostengono 38 Cfu. In base a questa media sono state stabilite in modo empirico le suddette 3 classi. Abbiamo
fatto in modo inoltre di dare una valenza alla distribuzione del dataset. Infatti circa 3000 studenti provengono dallo scentifico.
Abbiamo normalizzato le medie dei CFU aggiungendo o sottraendo CFU alla media in base a quanta era la percentuale di presenza
di una determinata maturità all'interno del dataset. Nonostante le maturità siano "ben suddivise" l'approccio utilizzato è puramente
empirico e privo di qualsiasi fondamento scientifico.
"""

import json
import csv

School = json.load(open("DictSchool.txt"))

scuole = []

for item in School:
    scuola = School.get(item)
    scuola = scuola[1:len(scuola)-1]
    scuole.append(scuola)

laureati = [tuple(row) for row in csv.reader(open('DatasetTriennali.csv', 'r'))]
tot = 0
count = 0
average = 0
scuola_media = []
j = 0
percent = 0
new_ave = 0
for i in range(0,len(scuole) - 1):
    tot = 0
    count = 0
    for j in range(0, len(laureati)):
        key = scuole[i]
        item = laureati[j][13]
        item = item[1:len(item) - 1]
        if item == key[0:len(key)]:
            primo = int(laureati[j][2])
            if primo != -1 and primo <= 60:
               tot = tot + primo
               count = count + 1
    average = tot / count
    average = round(average)
    percent = (count/3604) * 100
    print(key, "ha la media di: ", average, "e rappresenta il ", percent, "% del Dataset")
    coeff = (round(percent/10))
    if coeff < 0.6:
        new_ave = average - 2
    else:
        new_ave = average + (round(percent/10))
    scuola_media.append(new_ave)

print(scuola_media)