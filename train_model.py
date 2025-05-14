import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Each row: [moisture, temp, absorb1, absorb2, pressure, humidity, carbonate], label
data = [
    [12.5, 23.0, 0.12, 0.15, 1001, 30.0, 1, "Sandy"],
    [12.6, 23.1, 0.13, 0.16, 1000, 30.5, 1, "Sandy"],
    [14.0, 25.5, 0.10, 0.13, 1005, 33.0, 0, "Clay"],
    [14.1, 25.6, 0.11, 0.12, 1004, 32.8, 0, "Clay"],
    [18.0, 20.0, 0.08, 0.09, 990, 45.0, 1, "Loamy"],
    [18.1, 20.2, 0.09, 0.10, 989, 45.2, 1, "Loamy"],
    [20.0, 22.0, 0.20, 0.22, 980, 35.0, 0, "Silty"],
    [20.2, 22.1, 0.21, 0.23, 979, 35.5, 0, "Silty"],
    [16.5, 27.0, 0.14, 0.16, 995, 29.0, 1, "Peaty"],
    [16.6, 27.1, 0.13, 0.17, 994, 28.8, 1, "Peaty"]
]

df = pd.DataFrame(data, columns=["moisture", "temp", "absorb1", "absorb2", "pressure", "humidity", "carbonate", "soil_type"])

X = df.drop("soil_type", axis=1)
y = df["soil_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

with open("soil_model.pkl", "wb") as f:
    pickle.dump(model, f)
