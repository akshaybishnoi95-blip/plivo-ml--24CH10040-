import pandas as pd

df = pd.read_csv("eot_data/english/labels.csv")
print(df.head(10))
print(df["label"].value_counts())
print(df["turn_id"].nunique(), "unique turns")