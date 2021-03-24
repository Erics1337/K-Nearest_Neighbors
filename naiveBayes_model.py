from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")

  
categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']    #convert from catagorical to numerical

for cat in categorical:
    categories = pd.Categorical(df[cat])
    categories2 = pd.Categorical(dfTest[cat])
    df[cat] = categories.codes
    dfTest[cat] = categories2.codes


XTrain = df["education"]
X = XTrain.to_numpy()
X = np.reshape(X, (26049,1))
yTrain = df["income"]
y = yTrain.to_numpy()
XTest = dfTest["education"]
XTest = XTest.to_numpy()
XTest = np.reshape(XTest, (6512,1))


gnb = GaussianNB()
yp = gnb.fit(X, y)
yp = yp.predict(XTest) 

print(yp)
pd.DataFrame(yp).to_csv('Swanson.csv', header=None, index=None)
