import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


df = pd.read_csv("data/student-por.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

plt.hist(df["G3"])
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Grades")
plt.show()

print(df["G3"].describe())
print(df.isnull().sum())

print(df.select_dtypes(include="number").columns)

print(df.select_dtypes(include="number").describe())

numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()
print(correlation["G3"])

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))

plt.scatter(df["G1"], df["G3"])

plt.xlabel("First Period Grade (G1)")
plt.ylabel("Final Grade (G3)")
plt.title("G1 vs G3")

plt.show()

X = df.drop("G3", axis=1)
y = df["G3"]

print("Feature:", X.shape)
print("Target:", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


print(X.dtypes)

print(X.dtypes)


print(X.select_dtypes(include="object").columns)

X_encoded = pd.get_dummies(X,drop_first=True)

print("Before encoding:", X_encoded.shape)
print("After encoding:", X_encoded.shape)

print("X shape:", X.shape)
print("X encoded shape:", X_encoded.shape)
print(X.dtypes)

print(X.shape)
print(X_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "student_performance_model.pkl")
print("Model saved successfully")

y_pred = model.predict(X_test)
print("Predicted values:")
print(y_pred[:10])
comparison = pd.DataFrame({
    "Actual G3":
y_test.values[:10],
    "Predicted G3": y_pred[:10]
})

print(comparison)


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R2:", r2)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred)
print("Random Forest MAE:", rf_mae)

rf_r2 = r2_score(y_test, rf_pred)
print("Random Forest R2:", rf_r2)


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Student Performance")

plt.show()











