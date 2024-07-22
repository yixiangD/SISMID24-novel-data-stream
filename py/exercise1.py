import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lm
# (a)
df = pd.read_csv("./data/MX_Dengue_trends.csv")
df["Date"] = pd.to_datetime(df["Date"])
# plt.show()

# (b)
df_train = df[:36]
model = lm()
model.fit(df_train[["dengue"]], df_train[["Dengue CDC"]])

# (c)
ymax = 50
a = model.coef_[0][0]
b = model.intercept_[0]
y_fit = a * np.linspace(0, ymax, 20) + b

plt.figure()
plt.scatter(df_train["dengue"], df_train["Dengue CDC"])
plt.plot(np.linspace(0, ymax, 20), y_fit, color="r")
# plt.show()

# (d)
df_valid = df[37:]
y_pred = df_valid[["dengue"]] * a + b
y_true = df_valid.iloc[:, 2]
err = sklearn.metrics.mean_squared_error(y_true, y_pred)
print(err)
plt.figure()
plt.scatter(df_valid["dengue"], df_valid["Dengue CDC"])
plt.plot(df_valid[["dengue"]], y_pred, color="r")
# plt.show()
