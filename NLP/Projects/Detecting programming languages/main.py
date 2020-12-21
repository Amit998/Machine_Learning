import pandas as pd

df=(pd.read_csv("Qustions.csv",nrows=1_000_000,encoding="ISO-8859-1",usecols=['Title','Id']))

title=[ _ for _ in df['Title']]