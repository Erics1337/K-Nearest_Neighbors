from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv("UserKnowledge-train.csv")
dfTest = pd.read_csv("UserKnowledge-test.csv")


X = df.drop([" UNS"], 1)
y = df[" UNS"]

knn = KNeighborsClassifier()
knn.fit(X,y)
yp = knn.predict(dfTest)

pd.DataFrame(yp).to_csv('Swanson.csv', header=None, index=None)

