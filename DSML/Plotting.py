import pandas as pd
import matplotlib.pyplot as plt

predictiveAttributeDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")

for i in range(0, len(predictiveAttributeDegree)):
    plt.scatter(predictiveAttributeDegree[i][2], predictiveAttributeDegree[i][11])

for i in range(0, len(predictiveAttributeNotDegree)):
    plt.scatter(predictiveAttributeNotDegree[i][2], predictiveAttributeNotDegree[i][11])


plt.show()