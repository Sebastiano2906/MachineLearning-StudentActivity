import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing


data = pd.read_csv("C:/Users/sebas/PycharmProjects/MachineLearning-Local/DSML/Dataset/DatasetTriennali.csv", encoding='utf-8')
le = preprocessing.LabelEncoder()

DataFrameRichiesta = pd.DataFrame(
    {
        "1": data["1"],
        "VOTO_DIPLOMA": data["VOTO_DIPLOMA"],
        "TIPO_MATURITA": le.fit_transform(data["TIPO_MATURITA"])
    }
)

DataFrameEsteso = pd.DataFrame(
    {
        "Matr": le.fit_transform(data["Matr"]),
        "CF": le.fit_transform(data["CF"]),
        "1": data["1"],
        "Coorte": data["Coorte"],
        "Anno_Diploma": data["ANNO_DIPLOMA"],
        "VOTO_DIPLOMA": data["VOTO_DIPLOMA"],
        "CODICE_MECCANOGRAFICO": le.fit_transform(data["CODICE_MECCANOGRAFICO"]),
        "TIPO_MATURITA": le.fit_transform(data["TIPO_MATURITA"]),
        "STATO_STUDENTE": le.fit_transform(data["STATO_STUDENTE"]),
        "MOTIVO_STATO_STUDENTE": le.fit_transform(data["MOTIVO_STATO_STUDENTE"]),
    }
)

plt.matshow(DataFrameRichiesta.corr())
plt.xticks(range(len(DataFrameRichiesta.columns)), DataFrameRichiesta.columns)
plt.yticks(range(len(DataFrameRichiesta.columns)), DataFrameRichiesta.columns)
plt.colorbar()
plt.show()


plt.matshow(DataFrameEsteso.corr())
plt.xticks(range(len(DataFrameEsteso.columns)), DataFrameEsteso.columns, rotation=45, fontsize=6)
plt.yticks(range(len(DataFrameEsteso.columns)), DataFrameEsteso.columns)
plt.colorbar()
plt.show()