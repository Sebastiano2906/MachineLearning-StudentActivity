import os
import pandas as pd
import csv
import json


newData = [tuple(row) for row in csv.reader(open('C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/Dataset/newdata.csv', 'r'))]
tipoMat = json.load(open("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/Tipo_mat.txt"))
student = []
studentTemp = []
for i in range(1, len(newData)):
    cfu = newData[i][8]
    if cfu != '':
        cfu = int(cfu)
        id = int(newData[i][0])
        school = int(tipoMat.get(newData[i][4]))
        voto_dip = int(newData[i][3])

        studentTemp = [id, school, voto_dip, cfu]
        student.append(studentTemp)

with open('C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/FileGenerated/newData.txt', 'w') as file:
    file.write(json.dumps(student))