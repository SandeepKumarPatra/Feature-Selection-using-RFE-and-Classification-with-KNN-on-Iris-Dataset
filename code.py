import pandas as pd
import numpy as np

df = pd.read_csv("Iris.csv")

# Features and target
x = df.drop(columns=["Species", "Id"])
y = df["Species"]

# Feature Selection (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)

rfe = RFE(model, n_features_to_select=2)

rfe.fit(x, y)

selected_features = x.columns[rfe.support_]

print("Selected Features:", selected_features)

# Train Test Split

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Scaling

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

ss = StandardScaler()
le = LabelEncoder()

X_scale_train = ss.fit_transform(xtrain)
X_scale_test = ss.transform(xtest)

y_scale_train = le.fit_transform(ytrain)
y_scale_test = le.transform(ytest)

# KNN Model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_scale_train, y_scale_train)

pred = model.predict(X_scale_test)

print("Predictions:", pred)
