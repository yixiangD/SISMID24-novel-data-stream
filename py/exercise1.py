import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lm
# (a)
df = pd.read_csv("./data/MX_Dengue_trends.csv")
df["Date"] = pd.to_datetime(df["Date"])
print(df)
# plt.show()

# (b)
df_train = df[:36]
model = lm()
model.fit(df_train[["dengue"]], df_train[["Dengue CDC"]])
print(model.coef_)
print(model.intercept_)

# (c)
ymax = 50
a = model.coef_[0][0]
b = model.intercept_[0]
y_fit = a * np.linspace(0, ymax, 20) + b
plt.figure()
plt.scatter(df_train["dengue"], df_train["Dengue CDC"])
plt.plot(np.linspace(0, ymax, 20), y_fit, color="r")
plt.show()

# (d)
df_valid = df[36:]
y_pred = df_valid[["dengue"]] * a + b
plt.figure()
plt.scatter(df_valid["dengue"], df_valid["Dengue CDC"])
plt.plot(df_valid[["dengue"]], y_pred, color="r")
plt.show()
