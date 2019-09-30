import pandas as pd
import os
import matplotlib.pyplot as plt

def load_housing_data(housing_path="Dataset"):

    csv_path = os.path.join(housing_path, "DatasetStudenti.csv")
    return pd.read_csv(csv_path)

studenti = load_housing_data()

print("Stampa delle prime 5 righe: \n {0}".format(studenti.head()))
print("Stampa info: \n {0}".format(studenti.info()))


#studenti.hist(bins=50, figsize=(20,15))
#plt.show()

corr_matrix = studenti.corr()

print("\nStampa delle correlazionni con attrubuto Coorte: \n {0}".format(corr_matrix["Coorte"]))