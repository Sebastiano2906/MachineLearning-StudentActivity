import os
import pandas as pd
import csv
import json


newData = [tuple(row) for row in csv.reader(open('../MachineLearning-StudentActivity/DSML/Dataset/newdata.csv', 'r'))]
tipoMat = json.load(open("../MachineLearning-StudentActivity/DSML/FileGenerated/Tipo_mat.txt"))
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

with open('../MachineLearning-StudentActivity/DSML/Dataset/newdata.csv', 'w') as file:
    file.write(json.dumps(student))