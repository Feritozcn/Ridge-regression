from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("read.csv")
X = df[["info1", "info2"]]
y = df["results"]
X_standart = (X - X.mean()) / X.std()
X_standart.insert(0, "intercept", 1)
X_train, X_test, y_train, y_test = train_test_split(X_standart, y, test_size=0.2, random_state=42)
X_T_train = X_train.T
alpha = 2
I = np.identity(X_train.shape[1])
I[0][0] = 0
penalty = alpha * I
coeffcients = np.linalg.inv(X_T_train @ X_train + penalty) @ X_T_train @ y_train
y_train_pred = np.dot(X_train.values, coeffcients)
y_test_pred = np.dot(X_test.values, coeffcients)
## using np.dot to multiply values to matching coefficients
plt.scatter(X_train["info1"], y_train, color='blue', label='real data') 
plt.scatter(X_train["info1"], y_train_pred, color='red', label='predicted values')  
plt.plot(X_train["info1"], np.dot(X_train.values, coeffcients), color='green', linestyle='--', label='Model slope')  
plt.title('real data vs prediction')
plt.xlabel('info1')
plt.ylabel('results')
plt.show()


