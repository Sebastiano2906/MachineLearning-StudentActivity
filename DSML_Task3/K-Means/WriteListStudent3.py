import os
import csv
import json


""" -------------------------genera file Liststudenti----------------------------"""


laureati = [tuple(row) for row in csv.reader(open(os.path.abspath("../../DSML/Dataset/DatasetTriennali.csv"), 'r'))]
tip_mat= json.load(open(os.path.abspath('../../DSML/FileGenerated/Tipo_mat.txt')))


Maturita = []
Maturitatemp = []

for i in range(1, len(laureati)):


    matricola_studente=int(laureati[i][0])


    num=len(laureati[i][13])
    stringa=(laureati[i][13][1:num-1])
    tipo_maturita=tip_mat.get(stringa)

    voto_diploma = int(laureati[i][11])
    CFU_primo = int(laureati[i][2])

    if CFU_primo != -1:
       if CFU_primo <= 60:
        Maturitatemp = [[matricola_studente,tipo_maturita, voto_diploma, CFU_primo]]# da classe a tipo
        Maturita.append(Maturitatemp)


with open(os.path.abspath("ListStudent3.txt"), 'w') as file:
    file.write(json.dumps(Maturita))
