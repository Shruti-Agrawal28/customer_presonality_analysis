import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("prediction/marketing_campaign05052023__211047.csv")
print(accuracy_score(df["Clusters"], df.prediction))
