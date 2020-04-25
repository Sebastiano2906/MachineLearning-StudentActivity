import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt



predictiveAttributeDegree = pd.read_json("../MachineLearning-StudentActivity/DSML/DecisionTree/predictiveDegree.txt", orient='records', dtype=True,typ="series")
predictiveAttributeNotDegree = pd.read_json("../MachineLearning-StudentActivity/DSML/DecisionTree/predictiveNotDegree.txt", orient='records', dtype=True,typ="series")


train_set = []
test_set = []
train_result = []
test_result = []
count = 0
train_percent = (len(predictiveAttributeDegree)/100)*80
for i in range(len(predictiveAttributeDegree)):
    if count < train_percent:
        count = count + 1
        voto_dip = predictiveAttributeDegree[i][11]/10
        train_set.append([voto_dip, predictiveAttributeDegree[i][13]])
        cfu_primo = (predictiveAttributeDegree[i][2]/10)
        train_result.append([cfu_primo])
    else:
        voto_dip = predictiveAttributeDegree[i][11]/10
        test_set.append([voto_dip, predictiveAttributeDegree[i][13]])
        cfu_primo = (predictiveAttributeDegree[i][2] / 10)
        test_result.append([cfu_primo])
train_percent = (len(predictiveAttributeNotDegree)/100)*80
count = 0
for i in range(len(predictiveAttributeNotDegree)):
    if count < train_percent:
        count = count + 1
        voto_dip = predictiveAttributeNotDegree[i][11]/10
        train_set.append([voto_dip, predictiveAttributeNotDegree[i][13]])
        cfu_primo = (predictiveAttributeNotDegree[i][2] / 10)
        train_result.append([cfu_primo])
    else:
        voto_dip = predictiveAttributeNotDegree[i][11]/10
        test_set.append([voto_dip, predictiveAttributeNotDegree[i][13]])
        cfu_primo = (predictiveAttributeNotDegree[i][2] / 10)
        test_result.append([cfu_primo])

model = LinearRegression()
model.fit(train_set,train_result)

print("R2Score: ", model.score(test_set, test_result))
import math
from sklearn.metrics import mean_absolute_error
def metrics(m,X,y):
    yhat = m.predict(X)
    SS_Residual = sum((y-yhat)**2)
    SS_Total = sum((y-np.mean(y))**2)
    Scarto_totale = sum(abs(yhat-y))
    Scarto_totale_quadratico = sum((yhat-y)**2)
    mae = Scarto_totale / len(y)
    mse = Scarto_totale_quadratico / len(y)
    rmse = math.sqrt(mse)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adj_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-2 -1)
    return r_squared, adj_r_squared, mae, mse, rmse

r_sq, adj_r_squared, meanAbsoluteError, MSE, RMSE = metrics(model, test_set, test_result)
print("Adjusted_R2_Score: ", adj_r_squared)
print("R2_Score: ", r_sq)
print("MAE: ", meanAbsoluteError)
print("MSE: ", MSE)
print("RMSE: ", RMSE)
newStudent = [[11.1,2]]
print("Predicted: ", model.predict(newStudent))

def plot_learning_curves(model, X_train,X_val, y_train, y_val):
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="errore_train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="errore_test")
    plt.legend(numpoints=5)
    plt.show()


plot_learning_curves(model,train_set,test_set,train_result,test_result)