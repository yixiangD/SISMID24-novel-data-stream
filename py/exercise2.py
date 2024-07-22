import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lm
import sklearn
# (a)
df = pd.read_csv("./data/MX_Dengue_trends.csv")
df["Date"] = pd.to_datetime(df["Date"])
# print(df)
# plt.show()

y_pred = []
model = lm()
for i in range(len(df) - 36):
    df_train = df[i:i+36]
    model.fit(df_train[["dengue"]], df_train[["Dengue CDC"]])
    a = model.coef_[0][0]
    b = model.intercept_[0]
    if i == 0:
        a0 = a
        b0 = b
    x = df.iloc[i+36, 2]
    y_pred.append(a*x + b)
    # print(a*x+b, df.iloc[i+36, 1])
y_true = df.iloc[36:, 1]
err_dyn = sklearn.metrics.mean_squared_error(y_true, y_pred)
err_stat = sklearn.metrics.mean_squared_error(y_true, df.iloc[36:, 2] * a0 + b0)
print(err_dyn, err_stat)
plt.figure()
plt.plot(df.iloc[36:, 0], df.iloc[36:, 2] * a0 + b0, label="static")
plt.plot(df.iloc[36:, 0], y_true, label="true")
plt.plot(df.iloc[36:, 0], y_pred, label="dynamic")
plt.legend()
plt.show()
